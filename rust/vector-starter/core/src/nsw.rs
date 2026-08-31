use crate::{Dataset, Metric, Neighbor, Result, VectorError, VectorIndex, graph::search_layer, graph::prune_neighbors};

#[derive(Debug, Clone, Copy)]
pub struct NswConfig {
    pub max_connections: usize,
    pub ef_construction: usize,
    pub ef_search: usize,
}

impl Default for NswConfig {
    fn default() -> Self {
        Self {
            max_connections: 12,
            ef_construction: 48,
            ef_search: 32,
        }
    }
}

#[derive(Debug, Clone)]
pub struct NswIndex {
    dataset: Dataset,
    metric: Metric,
    config: NswConfig,
    adjacency: Vec<Vec<usize>>,
    entry_point: usize,
}

impl NswIndex {
    pub fn try_new(dataset: Dataset, metric: Metric, config: NswConfig) -> Result<Self> {
        //todo!("Chapter 3: insert rows into a bounded reciprocal proximity graph")
        dataset.validate_for_metric(metric)?;
        validate_config(&config)?;

        let mut adjacency: Vec<Vec<usize>> = Vec::<Vec<usize>>::with_capacity(dataset.len());

        for row in 0..dataset.len() {
            adjacency.push(Vec::new());
            if row == 0 {
                continue;
            }

            let candidates = search_layer(
                &dataset,
                metric,
                dataset.vector(row),
                &adjacency,
                &[0],
                config.ef_construction,
                row,
            );
            adjacency[row].extend(candidates.iter().map(|neighbor| neighbor.row));
            for &neighbor in &candidates {
                adjacency[neighbor.row].push(row);
            }
            prune_neighbors (
                &dataset,
                metric,
                row,
                &mut adjacency[row],
                config.max_connections,
            );
        }


        Ok(Self { 
            dataset, 
            metric, 
            config, 
            adjacency, 
            entry_point: 0 
        })
    }

    pub fn adjacency(&self) -> &[Vec<usize>] {
        &self.adjacency
    }

    pub fn search_with_ef(
        &self,
        query: &[f32],
        k: usize,
        ef_search: usize,
    ) -> Result<Vec<Neighbor>> {
        //todo!("Chapter 3: search the NSW graph with ef_search.max(k)")
        let mut result = search_layer(
            &self.dataset,
            self.metric,
            query,
            &self.adjacency,
            &[0],
            ef_search.max(k),
            self.adjacency.len(),
        );
        result.truncate(k);
        Ok(result)
    }
}

impl VectorIndex for NswIndex {
    fn kind(&self) -> &'static str {
        "nsw"
    }

    fn dataset(&self) -> &Dataset {
        &self.dataset
    }

    fn metric(&self) -> Metric {
        self.metric
    }

    fn search(&self, query: &[f32], k: usize) -> Result<Vec<Neighbor>> {
        self.search_with_ef(query, k, self.config.ef_search)
    }
}

fn validate_config(config: &NswConfig) -> Result<()> {
    if config.max_connections == 0 {
        return Err(VectorError::InvalidConfig(
            "max_connections must be greater than 0",
        ));
    }
    if config.ef_construction < config.max_connections {
        return Err(VectorError::InvalidConfig(
            "ef_construction must be greater than max_connections",
        ));
    }
    if config.ef_search == 0 {
        return Err(VectorError::InvalidConfig(
            "ef_search must be greater than 0",
        ));
    }
    Ok(())
}