#include "box_grasp_detection/box_detector.h"
#include <Eigen/Eigenvalues>
#include <pcl/ModelCoefficients.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/surface/convex_hull.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <numeric>
#include <queue>

namespace box_grasp_detection {
namespace {

struct SeedPatch {
    std::vector<int> indices;
    Eigen::Vector3f centroid = Eigen::Vector3f::Zero();
    Eigen::Vector3f normal = Eigen::Vector3f::UnitZ();
    float mean_height = -std::numeric_limits<float>::infinity();
    float score = -std::numeric_limits<float>::infinity();

    bool valid() const { return !indices.empty(); }
};

struct FaceExtractionResult {
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud;
    pcl::PointCloud<pcl::Normal>::Ptr normals;
    Eigen::Vector3f centroid = Eigen::Vector3f::Zero();
    Eigen::Vector3f normal = Eigen::Vector3f::UnitZ();
    float mean_height = -std::numeric_limits<float>::infinity();
    float score = -std::numeric_limits<float>::infinity();

    FaceExtractionResult()
        : cloud(new pcl::PointCloud<pcl::PointXYZ>),
          normals(new pcl::PointCloud<pcl::Normal>) {}

    bool valid() const { return !cloud->points.empty(); }
};

struct ClusterRoiMetrics {
    Eigen::Vector2f min_uv = Eigen::Vector2f::Constant(std::numeric_limits<float>::infinity());
    Eigen::Vector2f max_uv = Eigen::Vector2f::Constant(-std::numeric_limits<float>::infinity());
    float min_h = std::numeric_limits<float>::infinity();
    float max_h = -std::numeric_limits<float>::infinity();
};

struct PlanarRelation {
    float overlap_u = 0.0f;
    float overlap_v = 0.0f;
    float gap_u = 0.0f;
    float gap_v = 0.0f;
    float overlap_ratio_area = 0.0f;
    float overlap_ratio_u = 0.0f;
    float overlap_ratio_v = 0.0f;
};

float clampDot(float value) {
    return std::max(-1.0f, std::min(1.0f, value));
}

Eigen::Vector3f pointToEigen(const pcl::PointXYZ& point) {
    return Eigen::Vector3f(point.x, point.y, point.z);
}

Eigen::Vector3f orientNormalToReference(const pcl::Normal& normal, const Eigen::Vector3f& reference) {
    Eigen::Vector3f oriented(normal.normal_x, normal.normal_y, normal.normal_z);
    if (!std::isfinite(oriented.norm()) || oriented.norm() < 1e-6f) {
        return reference.normalized();
    }
    oriented.normalize();
    if (oriented.dot(reference) < 0.0f) {
        oriented = -oriented;
    }
    return oriented;
}

Eigen::Vector3f computeCentroid(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud,
    const std::vector<int>& indices) {
    Eigen::Vector3f centroid = Eigen::Vector3f::Zero();
    for (int idx : indices) {
        centroid += pointToEigen(cloud->points[idx]);
    }
    centroid /= static_cast<float>(indices.size());
    return centroid;
}

Eigen::Vector3f computeAverageNormal(
    const pcl::PointCloud<pcl::Normal>::Ptr& normals,
    const std::vector<int>& indices,
    const Eigen::Vector3f& reference) {
    Eigen::Vector3f avg = Eigen::Vector3f::Zero();
    for (int idx : indices) {
        avg += orientNormalToReference(normals->points[idx], reference);
    }
    if (avg.norm() < 1e-6f) {
        return reference.normalized();
    }
    avg.normalize();
    return avg;
}

Eigen::Vector3f chooseStableInPlaneReference(
    const Eigen::Vector3f& plane_normal,
    const Eigen::Vector3f& preferred_axis) {
    std::array<Eigen::Vector3f, 5> references = {
        preferred_axis.normalized(),
        Eigen::Vector3f::UnitX(),
        Eigen::Vector3f::UnitY(),
        Eigen::Vector3f(1.0f, 1.0f, 0.0f).normalized(),
        Eigen::Vector3f(1.0f, -1.0f, 0.0f).normalized()
    };

    float best_norm = -1.0f;
    Eigen::Vector3f best = Eigen::Vector3f::Zero();
    for (const auto& reference : references) {
        Eigen::Vector3f projected = reference - reference.dot(plane_normal) * plane_normal;
        float projected_norm = projected.norm();
        if (projected_norm > best_norm) {
            best_norm = projected_norm;
            best = projected;
        }
    }

    if (best_norm < 1e-4f) {
        best = plane_normal.unitOrthogonal();
    } else {
        best.normalize();
    }
    return best;
}

Eigen::Vector3f chooseAxisSignReference(
    const Eigen::Vector3f& axis,
    const Eigen::Vector3f& plane_normal) {
    std::array<Eigen::Vector3f, 4> references = {
        Eigen::Vector3f::UnitX(),
        Eigen::Vector3f::UnitY(),
        Eigen::Vector3f(1.0f, 1.0f, 0.0f).normalized(),
        Eigen::Vector3f(1.0f, -1.0f, 0.0f).normalized()
    };

    float best_alignment = -1.0f;
    Eigen::Vector3f best = plane_normal.unitOrthogonal();
    for (const auto& reference : references) {
        Eigen::Vector3f projected = reference - reference.dot(plane_normal) * plane_normal;
        if (projected.norm() < 1e-4f) {
            continue;
        }
        projected.normalize();
        const float alignment = std::fabs(axis.dot(projected));
        if (alignment > best_alignment) {
            best_alignment = alignment;
            best = projected;
        }
    }
    return best;
}

float absoluteProjectedAlignment(
    const Eigen::Vector3f& axis,
    const Eigen::Vector3f& plane_normal,
    const Eigen::Vector3f& reference) {
    Eigen::Vector3f projected = reference - reference.dot(plane_normal) * plane_normal;
    if (projected.norm() < 1e-4f) {
        return -1.0f;
    }
    projected.normalize();
    return std::fabs(axis.dot(projected));
}

Eigen::Vector3f orientAxisToReferences(
    const Eigen::Vector3f& axis,
    const Eigen::Vector3f& plane_normal,
    const std::array<Eigen::Vector3f, 3>& references) {
    Eigen::Vector3f oriented = axis;
    for (const auto& reference : references) {
        Eigen::Vector3f projected = reference - reference.dot(plane_normal) * plane_normal;
        if (projected.norm() < 1e-4f) {
            continue;
        }
        projected.normalize();
        const float dot = oriented.dot(projected);
        if (std::fabs(dot) > 1e-3f) {
            if (dot < 0.0f) {
                oriented = -oriented;
            }
            return oriented;
        }
    }
    return oriented;
}

ClusterRoiMetrics computeClusterRoiMetrics(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr& cluster,
    const Eigen::Vector3f& table_normal,
    const Eigen::Vector3f& plane_u,
    const Eigen::Vector3f& plane_v) {
    ClusterRoiMetrics metrics;
    for (const auto& pt : cluster->points) {
        const Eigen::Vector3f point = pointToEigen(pt);
        const Eigen::Vector2f uv(point.dot(plane_u), point.dot(plane_v));
        metrics.min_uv = metrics.min_uv.cwiseMin(uv);
        metrics.max_uv = metrics.max_uv.cwiseMax(uv);
        const float h = point.dot(table_normal);
        metrics.min_h = std::min(metrics.min_h, h);
        metrics.max_h = std::max(metrics.max_h, h);
    }
    return metrics;
}

float computeHeightGap(const ClusterRoiMetrics& lhs, const ClusterRoiMetrics& rhs) {
    if (lhs.max_h < rhs.min_h) {
        return rhs.min_h - lhs.max_h;
    }
    if (rhs.max_h < lhs.min_h) {
        return lhs.min_h - rhs.max_h;
    }
    return 0.0f;
}

PlanarRelation computePlanarRelation(
    const ClusterRoiMetrics& lhs,
    const ClusterRoiMetrics& rhs) {
    PlanarRelation relation;

    relation.overlap_u =
        std::min(lhs.max_uv.x(), rhs.max_uv.x()) -
        std::max(lhs.min_uv.x(), rhs.min_uv.x());
    relation.overlap_v =
        std::min(lhs.max_uv.y(), rhs.max_uv.y()) -
        std::max(lhs.min_uv.y(), rhs.min_uv.y());

    relation.gap_u = std::max(0.0f, -relation.overlap_u);
    relation.gap_v = std::max(0.0f, -relation.overlap_v);
    relation.overlap_u = std::max(0.0f, relation.overlap_u);
    relation.overlap_v = std::max(0.0f, relation.overlap_v);

    const float span_u_lhs = std::max(lhs.max_uv.x() - lhs.min_uv.x(), 1e-6f);
    const float span_u_rhs = std::max(rhs.max_uv.x() - rhs.min_uv.x(), 1e-6f);
    const float span_v_lhs = std::max(lhs.max_uv.y() - lhs.min_uv.y(), 1e-6f);
    const float span_v_rhs = std::max(rhs.max_uv.y() - rhs.min_uv.y(), 1e-6f);
    const float area_lhs = span_u_lhs * span_v_lhs;
    const float area_rhs = span_u_rhs * span_v_rhs;
    const float overlap_area = relation.overlap_u * relation.overlap_v;

    relation.overlap_ratio_area = overlap_area / std::min(area_lhs, area_rhs);
    relation.overlap_ratio_u = relation.overlap_u / std::min(span_u_lhs, span_u_rhs);
    relation.overlap_ratio_v = relation.overlap_v / std::min(span_v_lhs, span_v_rhs);
    return relation;
}

std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> mergeStackLikeClusters(
    const std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr>& clusters,
    const Eigen::Vector3f& table_normal,
    float voxel_leaf_size) {

    if (clusters.size() < 2) {
        return clusters;
    }

    const Eigen::Vector3f plane_u =
        chooseStableInPlaneReference(table_normal, Eigen::Vector3f(1.0f, 0.31f, 0.0f)).normalized();
    const Eigen::Vector3f plane_v = table_normal.cross(plane_u).normalized();

    std::vector<ClusterRoiMetrics> metrics;
    metrics.reserve(clusters.size());
    for (const auto& cluster : clusters) {
        metrics.push_back(computeClusterRoiMetrics(cluster, table_normal, plane_u, plane_v));
    }

    std::vector<int> parent(clusters.size());
    std::iota(parent.begin(), parent.end(), 0);

    const auto find_root = [&parent](int idx) {
        int root = idx;
        while (parent[root] != root) {
            root = parent[root];
        }
        while (parent[idx] != idx) {
            const int next = parent[idx];
            parent[idx] = root;
            idx = next;
        }
        return root;
    };

    const float overlap_ratio_threshold = 0.18f;
    const float min_overlap_extent = std::max(2.0f * voxel_leaf_size, 0.01f);
    const float height_gap_threshold = std::max(4.0f * voxel_leaf_size, 0.02f);
    const float side_gap_threshold = std::max(2.5f * voxel_leaf_size, 0.012f);

    for (size_t i = 0; i < clusters.size(); ++i) {
        for (size_t j = i + 1; j < clusters.size(); ++j) {
            const PlanarRelation relation = computePlanarRelation(metrics[i], metrics[j]);
            const bool strong_area_overlap =
                relation.overlap_u >= min_overlap_extent &&
                relation.overlap_v >= min_overlap_extent &&
                relation.overlap_ratio_area >= overlap_ratio_threshold;
            const bool side_contact_overlap =
                (relation.overlap_ratio_u >= 0.55f && relation.gap_v <= side_gap_threshold) ||
                (relation.overlap_ratio_v >= 0.55f && relation.gap_u <= side_gap_threshold);

            if (!strong_area_overlap && !side_contact_overlap) {
                continue;
            }

            const float height_gap = computeHeightGap(metrics[i], metrics[j]);
            if (height_gap > height_gap_threshold) {
                continue;
            }

            const int root_i = find_root(static_cast<int>(i));
            const int root_j = find_root(static_cast<int>(j));
            if (root_i != root_j) {
                parent[root_j] = root_i;
            }
        }
    }

    std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> merged_clusters;
    std::vector<int> root_to_index(clusters.size(), -1);

    for (size_t i = 0; i < clusters.size(); ++i) {
        const int root = find_root(static_cast<int>(i));
        if (root_to_index[root] < 0) {
            root_to_index[root] = static_cast<int>(merged_clusters.size());
            merged_clusters.emplace_back(new pcl::PointCloud<pcl::PointXYZ>);
        }

        pcl::PointCloud<pcl::PointXYZ>::Ptr& merged = merged_clusters[root_to_index[root]];
        merged->points.insert(merged->points.end(),
                              clusters[i]->points.begin(),
                              clusters[i]->points.end());
    }

    for (auto& cluster : merged_clusters) {
        cluster->width = cluster->points.size();
        cluster->height = 1;
        cluster->is_dense = true;
    }

    return merged_clusters;
}

std::vector<OrientedBoundingBox> suppressCoveredLowerFaces(
    const std::vector<OrientedBoundingBox>& boxes,
    const Eigen::Vector3f& table_normal,
    float voxel_leaf_size) {

    if (boxes.size() < 2) {
        return boxes;
    }

    const Eigen::Vector3f plane_u =
        chooseStableInPlaneReference(table_normal, Eigen::Vector3f(1.0f, 0.31f, 0.0f)).normalized();
    const Eigen::Vector3f plane_v = table_normal.cross(plane_u).normalized();

    std::vector<ClusterRoiMetrics> metrics;
    metrics.reserve(boxes.size());
    std::vector<float> center_heights;
    center_heights.reserve(boxes.size());

    for (const auto& box : boxes) {
        metrics.push_back(computeClusterRoiMetrics(box.cloud, table_normal, plane_u, plane_v));
        center_heights.push_back(box.center.dot(table_normal));
    }

    std::vector<bool> suppressed(boxes.size(), false);
    const float area_overlap_threshold = 0.12f;
    const float side_gap_threshold = std::max(2.5f * voxel_leaf_size, 0.012f);
    const float height_margin = std::max(2.0f * voxel_leaf_size, 0.008f);

    for (size_t hi = 0; hi < boxes.size(); ++hi) {
        for (size_t lo = 0; lo < boxes.size(); ++lo) {
            if (hi == lo || suppressed[lo]) {
                continue;
            }
            if (center_heights[hi] <= center_heights[lo] + height_margin) {
                continue;
            }

            const PlanarRelation relation = computePlanarRelation(metrics[hi], metrics[lo]);
            const bool strong_area_overlap = relation.overlap_ratio_area >= area_overlap_threshold;
            const bool side_contact_overlap =
                (relation.overlap_ratio_u >= 0.60f && relation.gap_v <= side_gap_threshold) ||
                (relation.overlap_ratio_v >= 0.60f && relation.gap_u <= side_gap_threshold);

            if (strong_area_overlap || side_contact_overlap) {
                suppressed[lo] = true;
            }
        }
    }

    std::vector<OrientedBoundingBox> filtered_boxes;
    filtered_boxes.reserve(boxes.size());
    for (size_t i = 0; i < boxes.size(); ++i) {
        if (!suppressed[i]) {
            filtered_boxes.push_back(boxes[i]);
        }
    }
    return filtered_boxes;
}

bool computeDominantHullAxis(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr& cluster,
    const Eigen::Vector3f& centroid,
    const Eigen::Vector3f& plane_normal,
    const Eigen::Vector3f& plane_u,
    const Eigen::Vector3f& plane_v,
    Eigen::Vector3f& axis_out) {

    if (!cluster || cluster->points.size() < 8) {
        return false;
    }

    pcl::PointCloud<pcl::PointXYZ>::Ptr projected_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    projected_cloud->points.reserve(cluster->points.size());
    for (const auto& pt : cluster->points) {
        Eigen::Vector3f point = pointToEigen(pt);
        point -= (point - centroid).dot(plane_normal) * plane_normal;
        projected_cloud->points.emplace_back(point.x(), point.y(), point.z());
    }
    projected_cloud->width = projected_cloud->points.size();
    projected_cloud->height = 1;
    projected_cloud->is_dense = true;

    pcl::ConvexHull<pcl::PointXYZ> hull;
    hull.setInputCloud(projected_cloud);
    hull.setDimension(2);

    pcl::PointCloud<pcl::PointXYZ>::Ptr hull_points(new pcl::PointCloud<pcl::PointXYZ>);
    std::vector<pcl::Vertices> polygons;
    hull.reconstruct(*hull_points, polygons);

    if (!hull_points || hull_points->points.size() < 3) {
        return false;
    }

    std::vector<int> ordered_indices;
    if (!polygons.empty() && polygons.front().vertices.size() >= 3) {
        ordered_indices.assign(polygons.front().vertices.begin(), polygons.front().vertices.end());
    } else {
        ordered_indices.resize(hull_points->points.size());
        std::iota(ordered_indices.begin(), ordered_indices.end(), 0);
    }

    if (ordered_indices.size() < 3) {
        return false;
    }

    double accum_cos = 0.0;
    double accum_sin = 0.0;
    double total_weight = 0.0;
    for (size_t i = 0; i < ordered_indices.size(); ++i) {
        const pcl::PointXYZ& p0 = hull_points->points[ordered_indices[i]];
        const pcl::PointXYZ& p1 = hull_points->points[ordered_indices[(i + 1) % ordered_indices.size()]];

        Eigen::Vector3f edge = pointToEigen(p1) - pointToEigen(p0);
        edge -= edge.dot(plane_normal) * plane_normal;
        const double edge_length = edge.norm();
        if (edge_length < 1e-4) {
            continue;
        }

        edge /= static_cast<float>(edge_length);
        const double theta = std::atan2(edge.dot(plane_v), edge.dot(plane_u));
        accum_cos += edge_length * std::cos(4.0 * theta);
        accum_sin += edge_length * std::sin(4.0 * theta);
        total_weight += edge_length;
    }

    if (total_weight < 1e-4) {
        return false;
    }

    const double concentration = std::hypot(accum_cos, accum_sin) / total_weight;
    if (concentration < 0.15) {
        return false;
    }

    const double dominant_theta = 0.25 * std::atan2(accum_sin, accum_cos);
    axis_out = (std::cos(dominant_theta) * plane_u + std::sin(dominant_theta) * plane_v).normalized();
    return axis_out.norm() > 1e-6f;
}

std::vector<float> computeHeights(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud,
    const Eigen::Vector3f& table_normal) {
    std::vector<float> heights(cloud->points.size(), 0.0f);
    for (size_t i = 0; i < cloud->points.size(); ++i) {
        heights[i] = pointToEigen(cloud->points[i]).dot(table_normal);
    }
    return heights;
}

std::vector<SeedPatch> buildSeedPatchCandidates(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr& cluster,
    const pcl::PointCloud<pcl::Normal>::Ptr& normals,
    const std::vector<float>& heights,
    const Eigen::Vector3f& table_normal,
    const pcl::search::KdTree<pcl::PointXYZ>::Ptr& tree,
    float voxel_leaf_size) {

    std::vector<SeedPatch> candidates;
    if (cluster->points.empty()) {
        return candidates;
    }

    const float max_height = *std::max_element(heights.begin(), heights.end());
    const float top_band = std::max(3.0f * voxel_leaf_size, 0.008f);
    const float seed_radius = std::max(2.5f * voxel_leaf_size, 0.012f);
    const float seed_plane_threshold = std::max(1.2f * voxel_leaf_size, 0.0035f);
    const float seed_normal_cos = std::cos(12.0f * static_cast<float>(M_PI) / 180.0f);
    const float min_table_alignment = 0.55f;
    const float curvature_threshold = 0.04f;
    const size_t min_seed_support = 8;

    for (size_t i = 0; i < cluster->points.size(); ++i) {
        if (heights[i] < max_height - top_band) {
            continue;
        }

        const pcl::Normal& seed_raw = normals->points[i];
        if (!std::isfinite(seed_raw.curvature) || seed_raw.curvature > curvature_threshold) {
            continue;
        }

        Eigen::Vector3f seed_normal = orientNormalToReference(seed_raw, table_normal);
        if (seed_normal.dot(table_normal) < min_table_alignment) {
            continue;
        }

        std::vector<int> neighbor_indices;
        std::vector<float> neighbor_distances;
        if (tree->radiusSearch(cluster->points[i], seed_radius, neighbor_indices, neighbor_distances) == 0) {
            continue;
        }

        SeedPatch patch;
        float height_sum = 0.0f;
        float curvature_sum = 0.0f;
        Eigen::Vector3f normal_sum = Eigen::Vector3f::Zero();

        const Eigen::Vector3f seed_point = pointToEigen(cluster->points[i]);
        for (int neighbor_idx : neighbor_indices) {
            if (heights[neighbor_idx] < max_height - top_band) {
                continue;
            }

            const pcl::Normal& neighbor_raw = normals->points[neighbor_idx];
            if (!std::isfinite(neighbor_raw.curvature) || neighbor_raw.curvature > curvature_threshold) {
                continue;
            }

            Eigen::Vector3f neighbor_normal = orientNormalToReference(neighbor_raw, table_normal);
            if (neighbor_normal.dot(seed_normal) < seed_normal_cos) {
                continue;
            }

            Eigen::Vector3f neighbor_point = pointToEigen(cluster->points[neighbor_idx]);
            float plane_distance = std::fabs((neighbor_point - seed_point).dot(seed_normal));
            if (plane_distance > seed_plane_threshold) {
                continue;
            }

            patch.indices.push_back(neighbor_idx);
            patch.centroid += neighbor_point;
            normal_sum += neighbor_normal;
            height_sum += heights[neighbor_idx];
            curvature_sum += neighbor_raw.curvature;
        }

        if (patch.indices.size() < min_seed_support) {
            continue;
        }

        patch.centroid /= static_cast<float>(patch.indices.size());
        patch.normal = normal_sum.normalized();
        if (patch.normal.dot(table_normal) < 0.0f) {
            patch.normal = -patch.normal;
        }
        patch.mean_height = height_sum / static_cast<float>(patch.indices.size());
        const float avg_curvature = curvature_sum / static_cast<float>(patch.indices.size());
        const float normal_alignment = patch.normal.dot(table_normal);
        patch.score = 4.0f * static_cast<float>(patch.indices.size()) +
                      400.0f * (patch.mean_height - (max_height - top_band)) +
                      20.0f * normal_alignment -
                      40.0f * avg_curvature;
        candidates.push_back(patch);
    }

    std::sort(candidates.begin(), candidates.end(),
              [](const SeedPatch& lhs, const SeedPatch& rhs) {
                  return lhs.score > rhs.score;
              });

    std::vector<SeedPatch> unique_candidates;
    const float dedup_distance = std::max(2.0f * voxel_leaf_size, 0.006f);
    for (const auto& candidate : candidates) {
        bool is_duplicate = false;
        for (const auto& kept : unique_candidates) {
            if ((candidate.centroid - kept.centroid).norm() < dedup_distance) {
                is_duplicate = true;
                break;
            }
        }
        if (!is_duplicate) {
            unique_candidates.push_back(candidate);
        }
        if (unique_candidates.size() >= 5) {
            break;
        }
    }

    return unique_candidates;
}

FaceExtractionResult growTopVisibleFace(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr& cluster,
    const pcl::PointCloud<pcl::Normal>::Ptr& normals,
    const std::vector<float>& heights,
    const Eigen::Vector3f& table_normal,
    const pcl::search::KdTree<pcl::PointXYZ>::Ptr& tree,
    const SeedPatch& seed,
    float voxel_leaf_size) {

    FaceExtractionResult result;
    if (!seed.valid()) {
        return result;
    }

    const float region_radius = std::max(3.0f * voxel_leaf_size, 0.014f);
    const float support_radius = std::max(2.5f * voxel_leaf_size, 0.011f);
    const float plane_threshold = std::max(1.5f * voxel_leaf_size, 0.0045f);
    const float normal_cos = std::cos(15.0f * static_cast<float>(M_PI) / 180.0f);
    const float curvature_threshold = 0.05f;
    const int min_local_support = 4;
    const size_t min_face_points = 30;

    std::vector<bool> in_region(cluster->points.size(), false);
    std::vector<bool> explored(cluster->points.size(), false);
    std::vector<int> region_indices;
    std::queue<int> queue;

    for (int idx : seed.indices) {
        if (idx < 0 || idx >= static_cast<int>(cluster->points.size()) || in_region[idx]) {
            continue;
        }
        in_region[idx] = true;
        explored[idx] = true;
        queue.push(idx);
        region_indices.push_back(idx);
    }

    while (!queue.empty()) {
        const int current_idx = queue.front();
        queue.pop();

        std::vector<int> neighbor_indices;
        std::vector<float> neighbor_distances;
        if (tree->radiusSearch(cluster->points[current_idx], region_radius, neighbor_indices, neighbor_distances) == 0) {
            continue;
        }

        for (int neighbor_idx : neighbor_indices) {
            if (neighbor_idx < 0 || neighbor_idx >= static_cast<int>(cluster->points.size()) || explored[neighbor_idx]) {
                continue;
            }
            explored[neighbor_idx] = true;

            const pcl::Normal& raw_normal = normals->points[neighbor_idx];
            if (!std::isfinite(raw_normal.curvature) || raw_normal.curvature > curvature_threshold) {
                continue;
            }

            Eigen::Vector3f neighbor_normal = orientNormalToReference(raw_normal, seed.normal);
            if (neighbor_normal.dot(seed.normal) < normal_cos) {
                continue;
            }

            const Eigen::Vector3f neighbor_point = pointToEigen(cluster->points[neighbor_idx]);
            const float plane_distance = std::fabs((neighbor_point - seed.centroid).dot(seed.normal));
            if (plane_distance > plane_threshold) {
                continue;
            }

            std::vector<int> support_indices;
            std::vector<float> support_distances;
            tree->radiusSearch(cluster->points[neighbor_idx], support_radius, support_indices, support_distances);

            int support_count = 0;
            for (int support_idx : support_indices) {
                const pcl::Normal& support_raw = normals->points[support_idx];
                if (!std::isfinite(support_raw.curvature) || support_raw.curvature > curvature_threshold) {
                    continue;
                }
                Eigen::Vector3f support_normal = orientNormalToReference(support_raw, seed.normal);
                if (support_normal.dot(seed.normal) < normal_cos) {
                    continue;
                }
                const Eigen::Vector3f support_point = pointToEigen(cluster->points[support_idx]);
                if (std::fabs((support_point - seed.centroid).dot(seed.normal)) > plane_threshold) {
                    continue;
                }
                ++support_count;
            }

            if (support_count < min_local_support) {
                continue;
            }

            in_region[neighbor_idx] = true;
            queue.push(neighbor_idx);
            region_indices.push_back(neighbor_idx);
        }
    }

    if (region_indices.size() < min_face_points) {
        return result;
    }

    Eigen::Vector3f refined_normal = computeAverageNormal(normals, region_indices, table_normal);
    Eigen::Vector3f refined_centroid = computeCentroid(cluster, region_indices);

    std::vector<int> refined_indices;
    refined_indices.reserve(region_indices.size());
    const float refine_plane_threshold = std::max(1.5f * voxel_leaf_size, 0.004f);
    const float refine_normal_cos = std::cos(12.0f * static_cast<float>(M_PI) / 180.0f);

    for (int idx : region_indices) {
        Eigen::Vector3f point = pointToEigen(cluster->points[idx]);
        Eigen::Vector3f point_normal = orientNormalToReference(normals->points[idx], refined_normal);
        if (point_normal.dot(refined_normal) < refine_normal_cos) {
            continue;
        }
        if (std::fabs((point - refined_centroid).dot(refined_normal)) > refine_plane_threshold) {
            continue;
        }
        refined_indices.push_back(idx);
    }

    if (refined_indices.size() < min_face_points) {
        return result;
    }

    result.normal = computeAverageNormal(normals, refined_indices, table_normal);
    result.centroid = computeCentroid(cluster, refined_indices);

    float height_sum = 0.0f;
    for (int idx : refined_indices) {
        result.cloud->points.push_back(cluster->points[idx]);
        result.normals->points.push_back(normals->points[idx]);
        height_sum += heights[idx];
    }

    result.cloud->width = result.cloud->points.size();
    result.cloud->height = 1;
    result.cloud->is_dense = true;
    result.normals->width = result.normals->points.size();
    result.normals->height = 1;
    result.normals->is_dense = true;
    result.mean_height = height_sum / static_cast<float>(refined_indices.size());
    result.score = static_cast<float>(refined_indices.size()) +
                   200.0f * result.mean_height +
                   10.0f * result.normal.dot(table_normal);
    return result;
}

}  // namespace

BoxDetector::BoxDetector()
    : voxel_leaf_size_(0.005),
      passthrough_min_z_(0.01),
      passthrough_max_z_(1.5),
      sor_mean_k_(50),
      sor_stddev_mul_(1.0),
      plane_distance_threshold_(0.01),
      plane_max_iterations_(1000),
      cluster_tolerance_(0.02),
      cluster_min_size_(30),
      cluster_max_size_(25000)
{
    ros::NodeHandle pnh("~");
    pnh.param("voxel_leaf_size", voxel_leaf_size_, voxel_leaf_size_);
    pnh.param("passthrough_min_z", passthrough_min_z_, passthrough_min_z_);
    pnh.param("passthrough_max_z", passthrough_max_z_, passthrough_max_z_);
    pnh.param("sor_mean_k", sor_mean_k_, sor_mean_k_);
    pnh.param("sor_stddev_mul", sor_stddev_mul_, sor_stddev_mul_);
    pnh.param("plane_distance_threshold", plane_distance_threshold_, plane_distance_threshold_);
    pnh.param("plane_max_iterations", plane_max_iterations_, plane_max_iterations_);
    pnh.param("cluster_tolerance", cluster_tolerance_, cluster_tolerance_);
    pnh.param("cluster_min_size", cluster_min_size_, cluster_min_size_);
    pnh.param("cluster_max_size", cluster_max_size_, cluster_max_size_);
}

std::vector<OrientedBoundingBox> BoxDetector::detectBoxes(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr& input_cloud) {
    
    std::vector<OrientedBoundingBox> boxes;
    
    if (!input_cloud || input_cloud->points.empty()) {
        ROS_WARN("Input cloud is empty!");
        return boxes;
    }
    
    // Step 1: Preprocess point cloud
    pcl::PointCloud<pcl::PointXYZ>::Ptr preprocessed = preprocessCloud(input_cloud);
    if (preprocessed->points.empty()) {
        ROS_WARN("Preprocessed cloud is empty!");
        return boxes;
    }
    
    // Step 2: Remove plane (table surface) & get Table Normal
    Eigen::Vector3f table_normal(0, 0, -1);
    pcl::PointCloud<pcl::PointXYZ>::Ptr objects_cloud = removePlane(preprocessed, table_normal);
    if (objects_cloud->points.empty()) {
        ROS_WARN("No objects found after plane removal!");
        return boxes;
    }

    // Step 3: Coarse clustering (temporary ROI proxy before YOLO-OBB ROI cropping)
    std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> coarse_clusters = extractClusters(objects_cloud);
    const size_t initial_cluster_count = coarse_clusters.size();
    coarse_clusters = mergeStackLikeClusters(coarse_clusters, table_normal, static_cast<float>(voxel_leaf_size_));
    ROS_INFO("Found %lu coarse ROIs after merge (%lu initial Euclidean clusters)",
             coarse_clusters.size(), initial_cluster_count);
    
    // Step 4: Extract the top-most visible single face from each coarse ROI
    for (const auto& cluster : coarse_clusters) {
        if (cluster->points.size() < 50) continue;

        // 4.1: Compute normals and curvatures inside the coarse ROI
        pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());
        tree->setInputCloud(cluster);
        
        pcl::NormalEstimationOMP<pcl::PointXYZ, pcl::Normal> ne;
        ne.setInputCloud(cluster);
        ne.setSearchMethod(tree);
        ne.setRadiusSearch(std::max(2.5 * voxel_leaf_size_, 0.012));  // keep normals local to one visible face
        pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
        ne.compute(*normals);

        // 4.2: Build top seed patches from the highest, flattest, most supported points
        std::vector<float> heights = computeHeights(cluster, table_normal);
        std::vector<SeedPatch> seed_candidates = buildSeedPatchCandidates(
            cluster, normals, heights, table_normal, tree, static_cast<float>(voxel_leaf_size_));

        if (seed_candidates.empty()) {
            ROS_DEBUG("Skipping coarse ROI: failed to find a stable top seed patch");
            continue;
        }

        // 4.3: Region-growing from the best seed patches, with planar consistency to avoid merging boxes
        FaceExtractionResult best_face;
        for (const auto& seed_candidate : seed_candidates) {
            FaceExtractionResult face = growTopVisibleFace(
                cluster, normals, heights, table_normal, tree, seed_candidate, static_cast<float>(voxel_leaf_size_));
            if (!face.valid()) {
                continue;
            }
            if (!best_face.valid() || face.score > best_face.score) {
                best_face = face;
            }
        }

        if (!best_face.valid()) {
            ROS_DEBUG("Skipping coarse ROI: top seed patches did not grow into a stable single face");
            continue;
        }

        // 4.4: Face pose generation
        OrientedBoundingBox obb = fitOBB(best_face.cloud, best_face.normal);
        
        // Filter by surface area (Length x Width)
        double area = obb.dimensions[0] * obb.dimensions[1];
        if (area >= 0.0004 && area <= 0.05) { // e.g., 4cm^2 to 500cm^2
            boxes.push_back(obb);
            float tilt_angle = std::acos(clampDot(best_face.normal.dot(table_normal))) * 180.0f / M_PI;
            ROS_INFO("Detected top visible face: %.3f x %.3f m, points=%zu, tilt=%.1f deg",
                     obb.dimensions[0], obb.dimensions[1], best_face.cloud->points.size(), tilt_angle);
        }
    }
    
    boxes = suppressCoveredLowerFaces(boxes, table_normal, static_cast<float>(voxel_leaf_size_));
    return boxes;
}

pcl::PointCloud<pcl::PointXYZ>::Ptr BoxDetector::preprocessCloud(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud) {
    
    pcl::PointCloud<pcl::PointXYZ>::Ptr filtered(new pcl::PointCloud<pcl::PointXYZ>);
    
    // Voxel grid downsampling
    pcl::VoxelGrid<pcl::PointXYZ> voxel;
    voxel.setInputCloud(cloud);
    voxel.setLeafSize(voxel_leaf_size_, voxel_leaf_size_, voxel_leaf_size_);
    voxel.filter(*filtered);
    
    // PassThrough filter (Z-axis)
    pcl::PassThrough<pcl::PointXYZ> pass;
    pass.setInputCloud(filtered);
    pass.setFilterFieldName("z");
    pass.setFilterLimits(passthrough_min_z_, passthrough_max_z_);
    pass.filter(*filtered);
    
    // Statistical outlier removal
    pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sor;
    sor.setInputCloud(filtered);
    sor.setMeanK(sor_mean_k_);
    sor.setStddevMulThresh(sor_stddev_mul_);
    sor.filter(*filtered);
    
    return filtered;
}

pcl::PointCloud<pcl::PointXYZ>::Ptr BoxDetector::removePlane(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud,
    Eigen::Vector3f& table_normal) {
    
    // RANSAC plane segmentation
    pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
    
    pcl::SACSegmentation<pcl::PointXYZ> seg;
    seg.setOptimizeCoefficients(true);
    seg.setModelType(pcl::SACMODEL_PLANE);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setMaxIterations(plane_max_iterations_);
    seg.setDistanceThreshold(plane_distance_threshold_);
    
    seg.setInputCloud(cloud);
    seg.segment(*inliers, *coefficients);
    
    if (inliers->indices.empty()) {
        ROS_WARN("No plane found in the cloud");
        return cloud;
    }
    
    // Extract table normal
    table_normal = Eigen::Vector3f(coefficients->values[0], coefficients->values[1], coefficients->values[2]).normalized();
    // Ensure it generally points towards the camera (-Z)
    if (table_normal.dot(Eigen::Vector3f(0, 0, -1)) < 0) {
        table_normal = -table_normal;
    }

    // Extract non-plane points
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_without_plane(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::ExtractIndices<pcl::PointXYZ> extract;
    extract.setInputCloud(cloud);
    extract.setIndices(inliers);
    extract.setNegative(true);  // Extract points NOT in the plane
    extract.filter(*cloud_without_plane);
    
    return cloud_without_plane;
}

std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> BoxDetector::extractClusters(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud) {
    
    std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> clusters;
    
    // KdTree for search
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
    tree->setInputCloud(cloud);
    
    // Euclidean clustering
    std::vector<pcl::PointIndices> cluster_indices;
    pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
    ec.setClusterTolerance(cluster_tolerance_);
    ec.setMinClusterSize(cluster_min_size_);
    ec.setMaxClusterSize(cluster_max_size_);
    ec.setSearchMethod(tree);
    ec.setInputCloud(cloud);
    ec.extract(cluster_indices);
    
    // Extract each cluster
    for (const auto& indices : cluster_indices) {
        pcl::PointCloud<pcl::PointXYZ>::Ptr cluster(new pcl::PointCloud<pcl::PointXYZ>);
        for (const auto& idx : indices.indices) {
            cluster->points.push_back(cloud->points[idx]);
        }
        cluster->width = cluster->points.size();
        cluster->height = 1;
        cluster->is_dense = true;
        clusters.push_back(cluster);
    }
    
    return clusters;
}

OrientedBoundingBox BoxDetector::fitOBB(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr& cluster,
    const Eigen::Vector3f& face_normal) {
    
    OrientedBoundingBox obb;
    obb.cloud = cluster;
    
    // Face centroid
    Eigen::Vector4f centroid_h;
    pcl::compute3DCentroid(*cluster, centroid_h);
    const Eigen::Vector3f centroid(centroid_h[0], centroid_h[1], centroid_h[2]);

    Eigen::Vector3f normal = face_normal.normalized();
    if (normal.dot(Eigen::Vector3f(0, 0, -1)) < 0.0f) {
        normal = -normal;
    }

    Eigen::Vector3f reference_in_plane = chooseStableInPlaneReference(normal, Eigen::Vector3f(1.0f, 0.3f, 0.0f));
    Eigen::Vector3f plane_u = reference_in_plane.normalized();
    Eigen::Vector3f plane_v = normal.cross(plane_u).normalized();

    Eigen::Matrix2f covariance = Eigen::Matrix2f::Zero();
    for (const auto& pt : cluster->points) {
        Eigen::Vector3f diff = pointToEigen(pt) - centroid;
        Eigen::Vector2f local(diff.dot(plane_u), diff.dot(plane_v));
        covariance += local * local.transpose();
    }
    covariance /= static_cast<float>(cluster->points.size());

    Eigen::SelfAdjointEigenSolver<Eigen::Matrix2f> eig_solver(covariance);
    Eigen::Vector2f eigen_values = eig_solver.eigenvalues();
    Eigen::Vector2f major_2d = eig_solver.eigenvectors().col(1);
    Eigen::Vector2f minor_2d = eig_solver.eigenvectors().col(0);

    Eigen::Vector3f major_axis = (major_2d[0] * plane_u + major_2d[1] * plane_v).normalized();
    Eigen::Vector3f minor_axis = (minor_2d[0] * plane_u + minor_2d[1] * plane_v).normalized();

    float min_major = std::numeric_limits<float>::infinity();
    float max_major = -std::numeric_limits<float>::infinity();
    float min_minor = std::numeric_limits<float>::infinity();
    float max_minor = -std::numeric_limits<float>::infinity();
    float min_normal = std::numeric_limits<float>::infinity();
    float max_normal = -std::numeric_limits<float>::infinity();

    for (const auto& pt : cluster->points) {
        Eigen::Vector3f diff = pointToEigen(pt) - centroid;
        const float major_coord = diff.dot(major_axis);
        const float minor_coord = diff.dot(minor_axis);
        const float normal_coord = diff.dot(normal);
        min_major = std::min(min_major, major_coord);
        max_major = std::max(max_major, major_coord);
        min_minor = std::min(min_minor, minor_coord);
        max_minor = std::max(max_minor, minor_coord);
        min_normal = std::min(min_normal, normal_coord);
        max_normal = std::max(max_normal, normal_coord);
    }

    float length = max_major - min_major;
    float width = max_minor - min_minor;
    float thickness = max_normal - min_normal;

    const float aspect_ratio = std::max(length, width) / std::max(std::min(length, width), 1e-6f);
    const float eigen_ratio = eigen_values[1] / std::max(eigen_values[0], 1e-6f);
    const bool is_square_like = (aspect_ratio < 1.18f) || (eigen_ratio < 1.12f);

    if (is_square_like) {
        const std::array<Eigen::Vector3f, 3> square_lock_refs = {
            Eigen::Vector3f(1.0f, 0.37f, 0.0f).normalized(),
            Eigen::Vector3f(-0.23f, 1.0f, 0.0f).normalized(),
            Eigen::Vector3f(1.0f, -0.19f, 0.0f).normalized()
        };

        Eigen::Vector3f hull_axis;
        if (computeDominantHullAxis(cluster, centroid, normal, plane_u, plane_v, hull_axis)) {
            major_axis = hull_axis;
            minor_axis = normal.cross(major_axis).normalized();

            min_major = std::numeric_limits<float>::infinity();
            max_major = -std::numeric_limits<float>::infinity();
            min_minor = std::numeric_limits<float>::infinity();
            max_minor = -std::numeric_limits<float>::infinity();

            for (const auto& pt : cluster->points) {
                Eigen::Vector3f diff = pointToEigen(pt) - centroid;
                const float major_coord = diff.dot(major_axis);
                const float minor_coord = diff.dot(minor_axis);
                min_major = std::min(min_major, major_coord);
                max_major = std::max(max_major, major_coord);
                min_minor = std::min(min_minor, minor_coord);
                max_minor = std::max(max_minor, minor_coord);
            }

            length = max_major - min_major;
            width = max_minor - min_minor;
            ROS_INFO_THROTTLE(0.5,
                "[FacePose] square-like face detected, using hull-edge orientation fallback | aspect=%.3f eig=%.3f",
                aspect_ratio, eigen_ratio);
        }

        // For near-square faces, "long" vs "short" is physically ambiguous.
        // Choose x/y by a lexicographic comparison against several skewed global references,
        // which removes the remaining 90-degree ambiguity at 45-degree-like views.
        bool swap_axes = false;
        for (const auto& reference : square_lock_refs) {
            const float major_alignment = absoluteProjectedAlignment(major_axis, normal, reference);
            const float minor_alignment = absoluteProjectedAlignment(minor_axis, normal, reference);
            if (major_alignment < 0.0f || minor_alignment < 0.0f) {
                continue;
            }
            if (std::fabs(major_alignment - minor_alignment) < 1e-3f) {
                continue;
            }
            swap_axes = minor_alignment > major_alignment;
            break;
        }
        if (swap_axes) {
            std::swap(major_axis, minor_axis);
            std::swap(length, width);
        }

        major_axis = orientAxisToReferences(major_axis, normal, square_lock_refs);
        minor_axis = normal.cross(major_axis).normalized();
    }

    if (!is_square_like && width > length) {
        std::swap(length, width);
        std::swap(major_axis, minor_axis);
    }

    if (!is_square_like) {
        Eigen::Vector3f sign_reference = chooseAxisSignReference(major_axis, normal);
        if (major_axis.dot(sign_reference) < 0.0f) {
            major_axis = -major_axis;
        }
    }

    minor_axis = normal.cross(major_axis).normalized();
    if (minor_axis.norm() < 1e-6f) {
        minor_axis = major_axis.unitOrthogonal();
    }

    Eigen::Matrix3f rotation;
    rotation.col(0) = major_axis;
    rotation.col(1) = minor_axis;
    rotation.col(2) = normal;
    if (rotation.determinant() < 0.0f) {
        rotation.col(1) = -rotation.col(1);
    }

    const float center_major = 0.5f * (min_major + max_major);
    const float center_minor = 0.5f * (min_minor + max_minor);
    const float center_normal = 0.5f * (min_normal + max_normal);
    obb.center = centroid +
                 center_major * rotation.col(0) +
                 center_minor * rotation.col(1) +
                 center_normal * rotation.col(2);

    obb.dimensions[0] = length;
    obb.dimensions[1] = width;
    obb.dimensions[2] = std::max(thickness, 0.02f);

    ROS_INFO_THROTTLE(0.5,
        "[FacePose] dims: %.1f x %.1f x %.1f cm (long x short x normal) | "
        "x=[%.2f,%.2f,%.2f] y=[%.2f,%.2f,%.2f] z=[%.2f,%.2f,%.2f]",
        obb.dimensions[0] * 100.0f, obb.dimensions[1] * 100.0f, obb.dimensions[2] * 100.0f,
        rotation.col(0)[0], rotation.col(0)[1], rotation.col(0)[2],
        rotation.col(1)[0], rotation.col(1)[1], rotation.col(1)[2],
        rotation.col(2)[0], rotation.col(2)[1], rotation.col(2)[2]);

    obb.orientation = Eigen::Quaternionf(rotation);
    obb.orientation.normalize();

    return obb;
}

} // namespace box_grasp_detection
