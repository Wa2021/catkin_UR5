#ifndef BOX_DETECTOR_H
#define BOX_DETECTOR_H

#include <ros/ros.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/features/moment_of_inertia_estimation.h>
#include <pcl/common/common.h>
#include <pcl/common/transforms.h>
#include <pcl/common/pca.h>
#include <vector>

namespace box_grasp_detection {

struct OrientedBoundingBox {
    Eigen::Vector3f center;
    Eigen::Vector3f dimensions;  // length, width, height
    Eigen::Quaternionf orientation;
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud;
    
    OrientedBoundingBox() : cloud(new pcl::PointCloud<pcl::PointXYZ>) {}
};

class BoxDetector {
public:
    BoxDetector();
    ~BoxDetector() = default;

    // Main detection pipeline
    std::vector<OrientedBoundingBox> detectBoxes(
        const pcl::PointCloud<pcl::PointXYZ>::Ptr& input_cloud);

private:
    // Point cloud preprocessing
    pcl::PointCloud<pcl::PointXYZ>::Ptr preprocessCloud(
        const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud);
    
    // Remove plane (table)
    pcl::PointCloud<pcl::PointXYZ>::Ptr removePlane(
        const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud,
        Eigen::Vector3f& table_normal);
    
    // Euclidean clustering
    std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> extractClusters(
        const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud);
    
    // OBB fitting using PCA
    OrientedBoundingBox fitOBB(
        const pcl::PointCloud<pcl::PointXYZ>::Ptr& cluster,
        const Eigen::Vector3f& table_normal);

    // Parameters
    double voxel_leaf_size_;
    double passthrough_min_z_;
    double passthrough_max_z_;
    int sor_mean_k_;
    double sor_stddev_mul_;
    
    // Plane segmentation
    double plane_distance_threshold_;
    int plane_max_iterations_;
    
    // Clustering
    double cluster_tolerance_;
    int cluster_min_size_;
    int cluster_max_size_;
    
};

} // namespace box_grasp_detection

#endif // BOX_DETECTOR_H
