use crate::{Dataset, Metric, Neighbor, Result, VectorError, VectorIndex, graph::greedy_search, graph::search_layer, graph::prune_neighbors};
use rand::Rng;
use crate::search::DeterministicRng;
#[derive(Debug, Clone, Copy)]
pub struct HnswConfig {
    pub max_connections: usize,
    pub ef_construction: usize,
    pub ef_search: usize,
    pub max_level: usize,
    pub seed: u64,
}

impl Default for HnswConfig {
    fn default() -> Self {
        Self {
            max_connections: 12,
            ef_construction: 64,
            ef_search: 40,
            max_level: 16,
            seed: 0x5eed,
        }
    }
}

#[derive(Debug, Clone)]
pub struct HnswIndex {
    dataset: Dataset,
    metric: Metric,
    config: HnswConfig,
    levels: Vec<usize>,
    layers: Vec<Vec<Vec<usize>>>,
    entry_point: usize,
    top_level: usize,
}

impl HnswIndex {
    pub fn try_new(dataset: Dataset, metric: Metric, config: HnswConfig) -> Result<Self> {
        //todo!("Chapter 4: assign seeded levels and build every included graph layer")
        dataset.validate_for_metric(metric)?;
        validate_config(&config)?;

        let mut rng = DeterministicRng::new(config.seed);
        let mut top_level = 0;
        let mut layers: Vec<Vec<Vec<usize>>> = Vec::new();
        let mut entry_point = 0;
        let mut levels = Vec::with_capacity(dataset.len());
        // 1. Create a seeded random number generator
        let mut level_max: usize = 0;
        for _ in 0..dataset.len() {
            let level = sample_level(&mut rng, config.max_level);
            // Assign the level to the row
            levels.push(level);
            if level_max < level {
                level_max = level;
            }
            // Build the graph layer by layer
        }
        
        for _ in 0..=level_max {
            // Build the graph layer by layer, starting from the top level down to level
            layers.push(vec![Vec::new(); dataset.len()]);
        }
        for row in 0..dataset.len() {
            let level = levels[row];
            let mut nearest_entry: usize = 0;
            // 如果当前向量的层比当前最高层高，那更新最高层和入口点
            if level > top_level {
                top_level = level;
                entry_point = row;
                nearest_entry = entry_point;
            }
            // 从当前层开始，向下遍历每一层，直到0层
            for current_level in (0..=level).rev() {
                let query = dataset.vector(row);
                // 如果当前层是0层，则使用ef_construction进行搜索
                if current_level == 0 {
                    let candidates = search_layer(
                        &dataset,
                        metric,
                        query,
                        &layers[current_level],
                        &[nearest_entry],
                        config.ef_construction,
                        row,
                    );
                    layers[current_level][row].extend(candidates.iter().map(|neighbor| neighbor.row));
                    for &neighbor in &candidates {
                        layers[current_level][neighbor.row].push(row);
                    }
                    prune_neighbors (
                        &dataset,
                        metric,
                        row,
                        &mut layers[current_level][row],
                        config.max_connections,
                    );
                }
                // 其他层使用greedy_search找到最近的入口点
                else {
                    nearest_entry = greedy_search(
                        &dataset,
                        metric,
                        query,
                        &layers[current_level],
                        nearest_entry,
                        row,
                    );

                    /*let candidates = search_layer(
                        &dataset,
                        metric,
                        query,
                        &layers[current_level],
                        &[nearest_entry],
                        config.ef_construction.max(config.max_connections),
                        row,
                    );
                    let selected = candidates
                        .iter()
                        .take(config.max_connections)
                        .map(|neighbor| neighbor.row)
                        .collect::<Vec<_>>();
                    layers[current_level][row].extend(&selected);
                    for &neighbor in &selected {
                        layers[current_level][neighbor].push(row);
                    }
                    prune_neighbors (
                        &dataset,
                        metric,
                        row,
                        &mut layers[current_level][row],
                        config.max_connections,
                    );*/
                }
            }
        }

        Ok(Self {
            dataset,
            metric,
            config,
            levels,
            layers,
            entry_point,
            top_level,
        })
    }

    pub fn levels(&self) -> &[usize] {
        &self.levels
    }

    pub fn top_level(&self) -> usize {
        self.top_level
    }

    pub fn search_with_ef(
        &self,
        _query: &[f32],
        _k: usize,
        _ef_search: usize,
    ) -> Result<Vec<Neighbor>> {
        todo!("Chapter 4: descend greedily and search layer zero with ef_search.max(k)")
    }

    pub fn layer(&self, level: usize) -> Option<&[Vec<usize>]> {
        self.layers.get(level).map(Vec::as_slice)
    }
}

impl VectorIndex for HnswIndex {
    fn kind(&self) -> &'static str {
        "hnsw"
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

fn validate_config(config: &HnswConfig) -> Result<()> {
    if config.max_connections == 0 {
        return Err(VectorError::InvalidConfig(
            "HNSW max_connections must be greater than zero",
        ));
    }
    if config.ef_construction == 0 {
        return Err(VectorError::InvalidConfig(
            "HNSW ef_construction must be greater than zero",
        ));
    }
    if config.ef_search == 0 {
        return Err(VectorError::InvalidConfig(
            "HNSW ef_search must be greater than zero",
        ));
    }
    if config.max_level == 0 {
        return Err(VectorError::InvalidConfig(
            "HNSW max_level must be greater than zero",
        ));
    }
    Ok(())
}

fn sample_level(rng: &mut DeterministicRng, max_level: usize) -> usize {
    let mut level = 0;
    while level < max_level {
        if rng.coin_flip() {
            level += 1;  // 正面：上升一层
        } else {
            break;       // 反面：停止
        }
    }
    level
}