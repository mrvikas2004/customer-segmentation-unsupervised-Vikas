# customer-segmentation-unsupervised-Vikas
# AI-Driven Customer Intelligence System
## Advanced Customer Segmentation Using Unsupervised Learning

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3-orange)
![Status](https://img.shields.io/badge/Status-Complete-green)

---

## Project Title
AI-Driven Customer Intelligence System for Strategic Business Decision Making

---

## Problem Statement
A retail company has collected 1,067,371 transactions across two years but lacks
the ability to identify:
- Who their most valuable customers are
- Which customers are likely to churn
- Which group spends the most
- Which group responds better to offers

No labels are provided. This project applies unsupervised machine learning to
discover hidden customer segments and convert them into actionable business strategies.

---

## Dataset
| Property        | Detail                                      |
|-----------------|---------------------------------------------|
| Name            | UCI Online Retail II                        |
| Source          | https://archive.ics.uci.edu/dataset/502     |
| Raw records     | 1,067,371 transactions                      |
| Customers       | 5,878 (after preprocessing)                 |
| Features built  | 11 engineered features                      |
| Period          | December 2009 – December 2011               |
| Country         | Primarily United Kingdom                    |

### Features Engineered
| Feature | Description |
|---|---|
| Recency | Days since last purchase |
| Frequency | Number of unique orders |
| Monetary | Total spend (£) |
| AvgOrderValue | Mean spend per order |
| TotalItems | Total quantity purchased |
| AvgItemsPerOrder | Average basket size |
| UniqueProducts | Number of distinct products |
| CustomerAge | Days since first purchase |
| AvgDaysBetweenOrders | Purchase cadence |
| SpendPerItem | Average price point preference |
| cancellation_rate | Proportion of cancelled orders |

---

## Algorithms Used

### 1. K-Means Clustering ✅ Winner
- Initialisation: k-means++
- Optimal K selected via Elbow Method + Silhouette Score + Davies-Bouldin Index
- Business-constrained K selection (K ≥ 3 enforced for actionability)
- **Result: K=6, Silhouette=0.2313, DBI=1.1238**

### 2. Hierarchical Clustering
- Linkage: Ward (minimises within-cluster variance)
- Dendrogram used to visually confirm K
- **Result: K=6, Silhouette=0.1535, DBI=1.2507**

### 3. DBSCAN
- eps tuned via k-distance graph (elbow method)
- Automatically detects noise/outlier customers
- **Result: 6 clusters, 15.4% noise, Silhouette=0.1649**

### 4. Gaussian Mixture Model (GMM)
- Component count selected via BIC score
- Soft probabilistic cluster assignments
- **Result: n=10 components, Silhouette=0.0635**

---

## Cluster Evaluation Methods
| Method | Purpose | Result |
|---|---|---|
| Elbow Method | Find inertia knee point | K=6 confirmed |
| Silhouette Score | Cluster separation quality | K-Means best (0.2313) |
| Davies-Bouldin Index | Cluster compactness ratio | K-Means best (1.1238) |
| BIC / AIC | GMM component selection | n=10 optimal |
| Weighted scoring | Final winner selection | K-Means wins |

---

## Dimensionality Reduction
| Method | Components | Variance Explained |
|---|---|---|
| PCA (2D) | 2 | 65.6% |
| PCA (3D) | 3 | 80.4% |
| PCA (95%) | 5 | 97.8% |
| t-SNE | 2 | Non-linear (bonus) |

---

## Key Results

### Pipeline Output