# ADR-002: KMeans for Customer Clustering

## Status

Accepted

## Context

The engine needs to segment customers into meaningful groups based on their behavior profiles. Requirements:

- Handle numeric features (revenue, frequency, recency)
- Produce interpretable clusters
- Scale to thousands of customers
- Support automatic cluster count selection

## Decision

Use **scikit-learn KMeans** for customer clustering with:

1. **Standardized features** - Z-score normalization before clustering
2. **Automatic k-selection** - Elbow method with silhouette validation
3. **Configurable k-range** - Default (3, 10) with override option

```python
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

class CustomerClusterer:
    def cluster(self, profiles: list[CustomerProfile]) -> ClusteringResult:
        features = self._extract_features(profiles)
        scaled = StandardScaler().fit_transform(features)

        if self.auto_select_k:
            k = self._select_k_elbow(scaled)

        kmeans = KMeans(n_clusters=k, random_state=self.seed)
        labels = kmeans.fit_predict(scaled)
        return self._create_segments(profiles, labels)
```

## Consequences

### Positive

- **Interpretable** - Cluster centers show average feature values
- **Fast** - O(n * k * iterations), suitable for 10k+ customers
- **Deterministic** - Fixed seed ensures reproducibility
- **Widely understood** - Team familiar with KMeans

### Negative

- **Assumes spherical clusters** - May miss complex cluster shapes
- **Sensitive to outliers** - Outliers can skew cluster centers
- **Requires feature scaling** - Must normalize before clustering

### Neutral

- K-selection heuristics (elbow, silhouette) are not perfect
- May need to experiment with different k values

## Alternatives Considered

### DBSCAN

Density-based clustering that finds arbitrary shapes. Rejected because:
- Requires careful epsilon/min_samples tuning
- May produce many noise points
- Less interpretable cluster definitions

### Hierarchical Clustering

Produces dendrogram of cluster relationships. Rejected because:
- O(n^2) memory complexity
- Slower for large datasets
- Overkill for our use case

### Gaussian Mixture Models

Probabilistic clustering with soft assignments. Rejected because:
- More complex to interpret
- Sensitive to initialization
- Overkill when hard assignments suffice

## References

- [scikit-learn KMeans](https://scikit-learn.org/stable/modules/clustering.html#k-means)
- [Elbow Method](https://en.wikipedia.org/wiki/Elbow_method_(clustering))
