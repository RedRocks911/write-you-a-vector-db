use std::{cmp::Reverse, vec};

use crate::{Dataset, Metric, Neighbor};

pub(crate) fn search_layer(
    dataset: &Dataset,
    metric: Metric,
    query: &[f32],
    adjacency: &[Vec<usize>],
    entry_points: &[usize],
    ef: usize,
    allowed_rows: usize,
) -> Vec<Neighbor> {
    //todo!("Chapter 3: traverse the graph with separate candidate and result frontiers")
    //候选集 (C)：初始化为一个空的最小堆（Min-Heap)
    let mut candidates: std::collections::BinaryHeap<Reverse<Neighbor>> = std::collections::BinaryHeap::new();
    //结果集 (R)：初始化为一个最大堆（Max-Heap),容量是ef_construction
    let mut results: std::collections::BinaryHeap<Neighbor> = std::collections::BinaryHeap::with_capacity(ef);

    let mut visited: Vec<bool> = (0..dataset.len())
        .map(|_| false)
        .collect();

    // 将入口点加入候选集和结果集
    for &entry in entry_points {
        if entry >= allowed_rows {
            continue;
        }
        let distance = metric.distance(dataset.vector(entry), query);
        candidates.push(Reverse(Neighbor {
            row: entry,
            distance: distance,
        }));
        results.push(Neighbor {
            row: entry,
            distance: distance,
        });
        visited[entry] = true;
    }

    while let Some(Reverse(checkpoint)) = candidates.pop(){ 
        /*//如果已经访问过了，那就跳过
        if visited[checkpoint.row] {
            continue;
        }
        visited[checkpoint.row] = true;*/
        // 如果候选集最好的那个距离都比结果集中最差的那个距离还要大，那就终止循环
        if checkpoint.distance > results.peek().unwrap().distance {
            break;
        }
        // 开始遍历所有checkpoint 的邻居，看是否能进候选集或者结果集
        for &neighbor_row in &adjacency[checkpoint.row] {
            // 如果已经访问过了，那就跳过
            if visited[neighbor_row] || neighbor_row >= allowed_rows {
                continue;
            }
            visited[neighbor_row] = true;
            let distance = metric.distance(dataset.vector(neighbor_row), query);
            let neighbor = Neighbor {
                row: neighbor_row,
                distance,
            };
            // 如果结果集的容量已经满了，并且当前候选点的距离大于结果集中的最大距离，则跳过该候选点
            // 如果结果集未满，则进入结果集和候选集
            if results.len() < results.capacity() {
                candidates.push(Reverse(neighbor));
                results.push(neighbor);
            }
            // 如果考察的点与查询点的距离小于结果集中最差的那个距离，则将其加入结果集，并从结果集中移除最差的那个点
            else if results.len() == results.capacity() {
                if neighbor.distance < results.peek().unwrap().distance {
                    candidates.push(Reverse(neighbor));
                    results.pop();
                    results.push(neighbor);
                }
                // 如果考察的点与查询点的距离等于结果集中最差的那个距离，则将其加入候选集，但不加入结果集
                else if neighbor.distance == results.peek().unwrap().distance {
                    candidates.push(Reverse(neighbor));
                }
            }
        }
    }

    results.into_iter().collect()
    
}

pub(crate) fn greedy_search(
    dataset: &Dataset,
    metric: Metric,
    query: &[f32],
    adjacency: &[Vec<usize>],
    entry: usize,
    allowed_rows: usize,
) -> usize {
    //todo!("Chapter 4: greedily descend one HNSW layer")
    let mut current = Neighbor {
        row: entry,
        distance: metric.distance(query, dataset.vector(entry)),
    };
    loop {
        let next = adjacency[current.row]
            .iter()
            .copied()
            .filter(|row| *row < allowed_rows)
            .map(|row| Neighbor {
                row,
                distance: metric.distance(query, dataset.vector(row)),
            })
            .min();
        match next {
            Some(next) if next < current => current = next,
            _ => return current.row,
        }
    }
}

pub(crate) fn prune_neighbors(
    dataset: &Dataset,
    metric: Metric,
    owner: usize,
    neighbors: &mut Vec<usize>,
    max_connections: usize,
) {
    //todo!("Chapter 3: retain the closest deterministic neighbor set")
    neighbors.sort_unstable();
    neighbors.dedup();
    neighbors.sort_unstable_by(|a, b|{
        let neighbor_a = Neighbor {
            row: *a,
            distance: metric.distance(dataset.vector(*a), dataset.vector(owner)),
        };
        let neighbor_b = Neighbor {
            row: *b,
            distance: metric.distance(dataset.vector(*b), dataset.vector(owner)),
        };
        neighbor_a.cmp(&neighbor_b)
    });

    neighbors.truncate(max_connections);

}
