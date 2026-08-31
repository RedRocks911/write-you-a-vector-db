use crate::{Dataset, Metric, Neighbor, Result, VectorIndex};
use crate::VectorError;
use crate::search::{DeterministicRng, TopK};
use rand::seq::SliceRandom; // 引入 shuffle 方法
use rand::thread_rng;        // 提供默认的线程级随机数生成器

#[derive(Debug, Clone, Copy)]
pub struct IvfFlatConfig {
    pub partitions: usize,
    pub probes: usize,
    pub iterations: usize,
    pub seed: u64,
}

impl Default for IvfFlatConfig {
    fn default() -> Self {
        Self {
            partitions: 16,
            probes: 4,
            iterations: 12,
            seed: 0x5eed,
        }
    }
}

#[derive(Debug, Clone)]
pub struct IvfFlatIndex {
    dataset: Dataset,
    metric: Metric,
    config: IvfFlatConfig,
    centroids: Vec<Vec<f32>>,
    lists: Vec<Vec<usize>>,
}

impl IvfFlatIndex {
    pub fn try_new(dataset: Dataset, metric: Metric, config: IvfFlatConfig) -> Result<Self> {
        //todo!("Chapter 2: train centroids and assign every row to an inverted list")
        dataset.validate_for_metric(metric)?;
        if config.partitions == 0 || config.partitions > dataset.len() {
            return Err(VectorError::InvalidConfig( 
                "partitions must be greater than 0 and less than or equal to dataset length", 
            ));
        }
        if config.probes == 0 || config.probes > config.partitions {
            return Err(VectorError::InvalidConfig(
                "probes must be greater than 0 and less than or equal to partitions", 
            ));  
        }
        if config.iterations == 0 {
            return Err(VectorError::InvalidConfig(
                "iterations must be greater than 0", 
            ));
        }
        if config.seed == 0 {
            return Err(VectorError::InvalidConfig(
                "seed must be greater than 0", 
            ));
        }
        let mut rng = DeterministicRng::new(config.seed);
        // 1. 创建一个包含所有行索引的数组
        let mut indices: Vec<usize> = (0..dataset.len()).collect();
        // 2. 打乱这个索引数组
        for end in (1..indices.len()).rev() {
            let selected = rng.index(end + 1);
            indices.swap(end, selected);
        }
        // 3. 选择前 config.partitions 个索引作为质心
        let mut centroids: Vec<Vec<f32>> = indices.into_iter()
            .take(config.partitions)
            .map(|idx| dataset.vector(idx).to_vec()) // 闭包参数改名为 idx，语义更清晰
            .collect();

        // 4. 初始化每个质心对应的列表
        let mut lists: Vec<Vec<usize>> = vec![Vec::new(); config.partitions];
        for _ in 0..config.iterations {
            let mut lists_new: Vec<Vec<usize>> = vec![Vec::new(); config.partitions];
            for (row_index, vector) in dataset.vectors().iter().enumerate() {
                let mut min_distance = f32::MAX;
                let mut list_index = 0;
                for (i, centroid) in centroids.iter_mut().enumerate() {
                    let distance = metric.distance(vector, centroid);
                    if distance < min_distance {
                        min_distance = distance;
                        list_index = i;
                    }
                }
                lists_new[list_index].push(row_index);
            }
            if lists_new == lists {
                break;
            }
            lists = lists_new;
            let dim = dataset.dimension();
            for (partition_id, row_indices) in lists.iter().enumerate() {
                if row_indices.is_empty() {
                    continue;
                }
                
                // 对每个维度分别求和
                let mut new_centroid = vec![0.0; dim];
                for d in 0..dim {
                    let sum: f32 = row_indices.iter()
                        .map(|&row_idx| dataset.vector(row_idx)[d])
                        .sum();
                    new_centroid[d] = sum / row_indices.len() as f32;
                }
                
                centroids[partition_id] = new_centroid;
            }
        }
        Ok(Self {
            dataset,
            metric,
            config,
            centroids,
            lists,
        })
    }

    pub fn centroids(&self) -> &[Vec<f32>] {
        &self.centroids
    }

    pub fn list_sizes(&self) -> Vec<usize> {
        self.lists.iter().map(Vec::len).collect()
    }

    pub fn search_with_probes(
        &self,
        query: &[f32],
        k: usize,
        probes: usize,
    ) -> Result<Vec<Neighbor>> {
        //todo!("Chapter 2: rank centroids, scan the selected lists, and keep top-k")
        if probes == 0 || probes >= self.config.partitions {
            return Err(VectorError::InvalidConfig( 
                "probes must be greater than 0 and less than partitions", 
            ));
        }
        // 1. 遍历所有质心，计算查询向量与每个质心的距离，生成 Neighbor 列表
        let mut centroid_neighbors: Vec<Neighbor> = self
            .centroids
            .iter()
            .enumerate()
            .map(|(idx, centroid)| {
                let distance = self.metric.distance(query, centroid);
                Neighbor {
                    row: idx,       // 记录质心的索引（即簇的 ID）
                    distance,       // 记录与查询向量的距离
                }
            })
            .collect();

        // 2. 按距离从小到大排序（最近优先）
        centroid_neighbors.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap());

        // 3. 截取前 `probes` 个质心
        // 注意：这里可以安全地 unwrap，因为前面已经校验过 probes < partitions
        let selected_probes: Vec<Neighbor> = centroid_neighbors
            .into_iter()
            .take(self.config.probes)
            .collect();

        // 4. 对每个选中的质心，扫描其对应的列表，计算查询向量与列表中每个向量的距离，并维护一个 TopK 结构来保存最近的 k 个邻居
        let mut top_k = TopK::new(k);
        for probe in &selected_probes {
            let list_id = probe.row; // 获取质心的索引，即列表的 ID
            for &row in &self.lists[list_id] {
                let vector = self.dataset.vector(row);
                let distance = self.metric.distance(query, vector);
                top_k.push(Neighbor { row, distance });
            }
        }
        Ok(top_k.into_sorted())
    }
}

impl VectorIndex for IvfFlatIndex {
    fn kind(&self) -> &'static str {
        "ivf_flat"
    }

    fn dataset(&self) -> &Dataset {
        &self.dataset
    }

    fn metric(&self) -> Metric {
        self.metric
    }

    fn search(&self, query: &[f32], k: usize) -> Result<Vec<Neighbor>> {
        self.search_with_probes(query, k, self.config.probes)
    }
}
