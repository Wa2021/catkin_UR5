#include "box_grasp_detection/box_detector.h"
#include <iomanip>
#include <iostream>
#include <memory>
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>
#include "box_grasp_detection/BoxGrasp.h"
#include "box_grasp_detection/BoxGraspArray.h"
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/common/centroid.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/common/pca.h>
#include <pcl/common/common.h>
#include <pcl/search/kdtree.h>

class BoxDetectorNode {
public:
    BoxDetectorNode() {
        ros::NodeHandle nh;
        ros::NodeHandle pnh("~");
        
        // Parameters
        pnh.param<std::string>("input_cloud_topic", input_cloud_topic_, "/camera/depth/color/points");
        
        // Publishers
        grasp_pub_ = nh.advertise<box_grasp_detection::BoxGraspArray>("box_grasps", 10);
        marker_pub_ = nh.advertise<visualization_msgs::MarkerArray>("box_markers", 10);
        object_cloud_pub_ = nh.advertise<sensor_msgs::PointCloud2>("detected_objects", 10);
        
        // Subscriber
        cloud_sub_ = nh.subscribe(input_cloud_topic_, 1, 
                                 &BoxDetectorNode::cloudCallback, this);
        
        // Initialize detector
        detector_ = std::make_shared<box_grasp_detection::BoxDetector>();
        
        ROS_INFO("Box Detector Node initialized");
        ROS_INFO("Subscribing to: %s", input_cloud_topic_.c_str());
        ROS_INFO("Publishing grasps to: box_grasps");
        ROS_INFO("Publishing markers to: box_markers");
    }
    
private:
    void cloudCallback(const sensor_msgs::PointCloud2ConstPtr& cloud_msg) {
        // Convert ROS cloud to PCL
        pcl::PointCloud<pcl::PointXYZ>::Ptr pcl_cloud(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::fromROSMsg(*cloud_msg, *pcl_cloud);
        
        // Start timing
        ros::Time start_time = ros::Time::now();

        // Detect boxes
        auto boxes = detector_->detectBoxes(pcl_cloud);
        
        // Generate Grasp List
        box_grasp_detection::BoxGraspArray grasp_array = generateGrasps(boxes, cloud_msg->header);
        
        // End timing
        double duration_ms = (ros::Time::now() - start_time).toSec() * 1000.0;
        // 使用 std::cout 强制输出到终端，避免 ROS 日志缓冲或过滤
        std::cout << "\033[1;32m[Performance] Detection Time: " << std::fixed << std::setprecision(2) << duration_ms << " ms\033[0m" << std::endl;

        if (boxes.empty()) {
            ROS_WARN_THROTTLE(5, "No boxes detected");

            return;
        }
        
        ROS_INFO("Detected %lu boxes", boxes.size());
        
        // Publish results
        grasp_pub_.publish(grasp_array);
        publishMarkers(boxes, cloud_msg->header);
        publishObjectClouds(boxes, cloud_msg->header);
    }
    
    box_grasp_detection::BoxGraspArray generateGrasps(const std::vector<box_grasp_detection::OrientedBoundingBox>& boxes,
                      const std_msgs::Header& header) {
        box_grasp_detection::BoxGraspArray grasp_array;
        grasp_array.header = header;
        
        for (const auto& box : boxes) {
            box_grasp_detection::BoxGrasp grasp;
            grasp.header = header;
            
            // Box dimensions — fitOBB 排序后语义：
            //   dimensions[0] = length  最长轴, col(0)
            //   dimensions[1] = width   次长轴, col(1) ← 夹爪开合方向（夹住次长边）
            //   dimensions[2] = height  最短轴, col(2) ← 夹爪接近/插入方向（物体最薄面法线）
            float box_length = box.dimensions[0];
            float box_width  = box.dimensions[1];  // 夹爪实际张开宽度
            float box_height = box.dimensions[2];  // 物体厚度（最短方向）

            grasp.length = box_length;
            grasp.width  = box_width;
            grasp.height = box_height;
            
            // Box pose
            grasp.box_pose.position.x = box.center[0];
            grasp.box_pose.position.y = box.center[1];
            grasp.box_pose.position.z = box.center[2];
            grasp.box_pose.orientation.x = box.orientation.x();
            grasp.box_pose.orientation.y = box.orientation.y();
            grasp.box_pose.orientation.z = box.orientation.z();
            grasp.box_pose.orientation.w = box.orientation.w();
            
            grasp.grasp_pose.header = header;
            
            // =====================================================
            // 顶部抓取位姿构建（从上方夹住短边）
            // 约定：夹爪 Z 轴 = 接近方向（从上向下）
            //        夹爪 X 轴 = 夹爪开合方向（沿 box 短边 col(1)）
            //        夹爪 Y 轴 = 夹爪手指排列方向（沿 box 长边 col(0)）
            //
            // 这样夹爪张开宽度 = box_width，保证夹住短边。
            // =====================================================
            Eigen::Matrix3f R = box.orientation.toRotationMatrix();

            // col(0)=最长轴，col(1)=次长轴（夹爪开合），col(2)=最短轴（接近方向）
            Eigen::Vector3f long_axis    = R.col(0);  // 最长轴
            Eigen::Vector3f short_axis   = R.col(1);  // 次长轴，夹爪开合
            Eigen::Vector3f approach_vec = R.col(2);  // 最短轴，物体最薄面法线

            // fitOBB 已保证 col(2) 朝向相机（-Z分量为正），
            // 再做防御：确保 approach_vec 朝向相机（dot with (0,0,-1) > 0）
            if (approach_vec.dot(Eigen::Vector3f(0, 0, -1)) < 0)
                approach_vec = -approach_vec;

            // 接近方向 = approach_vec 指向相机，夹爪从相机侧向下插入物体
            // 即夹爪 Z 轴 = -approach_vec（从相机侧朝向物体）
            Eigen::Vector3f grasp_z = -approach_vec;  // 夹爪插入方向（朝向物体）

            // 这里不再对 short_axis 做额外的 +X 半球翻转。
            // box.orientation 已经在 fitOBB 中稳定好每个轴的方向，
            // 直接沿用可以避免同一目标在抓取姿态层再发生 180 度来回跳变。
            Eigen::Vector3f grasp_x = short_axis;
            grasp_x -= grasp_x.dot(grasp_z) * grasp_z;
            if (grasp_x.norm() < 1e-6f) {
                grasp_x = long_axis - long_axis.dot(grasp_z) * grasp_z;
            }
            grasp_x.normalize();

            // 夹爪 Y 轴 = Z × X（右手系）
            Eigen::Vector3f grasp_y = grasp_z.cross(grasp_x).normalized();
            grasp_x = grasp_y.cross(grasp_z).normalized();

            // 保持 x 轴尽量与 box 的次长轴同向，保证夹爪开合方向稳定
            if (grasp_x.dot(short_axis) < 0.0f) {
                grasp_x = -grasp_x;
                grasp_y = -grasp_y;
            }

            // 构建旋转矩阵，保证右手系
            Eigen::Matrix3f grasp_rotation;
            grasp_rotation.col(0) = grasp_x;
            grasp_rotation.col(1) = grasp_y;
            grasp_rotation.col(2) = grasp_z;
            if (grasp_rotation.determinant() < 0)
                grasp_rotation.col(1) = -grasp_rotation.col(1);

            Eigen::Quaternionf grasp_quat(grasp_rotation);
            grasp_quat.normalize();

            // 抓取位置 = 顶面中心，对于薄片盒（只取顶面后），box.center基本就是顶面中心。
            // 向外偏移 approach_offset
            float approach_offset = 0.05f;
            Eigen::Vector3f grasp_pos = box.center + approach_vec * approach_offset;

            grasp.grasp_pose.pose.position.x = grasp_pos[0];
            grasp.grasp_pose.pose.position.y = grasp_pos[1];
            grasp.grasp_pose.pose.position.z = grasp_pos[2];
            grasp.grasp_pose.pose.orientation.x = grasp_quat.x();
            grasp.grasp_pose.pose.orientation.y = grasp_quat.y();
            grasp.grasp_pose.pose.orientation.z = grasp_quat.z();
            grasp.grasp_pose.pose.orientation.w = grasp_quat.w();
            
            sensor_msgs::PointCloud2 cloud_msg;
            pcl::toROSMsg(*box.cloud, cloud_msg);
            cloud_msg.header = header;
            grasp.object_cloud = cloud_msg;
            
            grasp.grasp_type = "top_grasp_short_edge";
            grasp.score = 0.8f;

            ROS_INFO("Grasp: L=%.3f W=%.3f H=%.3f | short_axis=[%.2f,%.2f,%.2f] | pos=[%.3f,%.3f,%.3f]",
                     box_length, box_width, box_height,
                     short_axis[0], short_axis[1], short_axis[2],
                     grasp.grasp_pose.pose.position.x,
                     grasp.grasp_pose.pose.position.y,
                     grasp.grasp_pose.pose.position.z);
            
            grasp_array.grasps.push_back(grasp);
        }
        
        return grasp_array;
    }
    
    // ... (后面的 publishMarkers 和 publishObjectClouds 都不用改) ...
    
    void publishMarkers(const std::vector<box_grasp_detection::OrientedBoundingBox>& boxes,
                       const std_msgs::Header& header) {
        visualization_msgs::MarkerArray marker_array;
        
        int id = 0;
        for (const auto& box : boxes) {
            // Box marker
            visualization_msgs::Marker box_marker;
            box_marker.header = header;
            box_marker.ns = "boxes";
            box_marker.id = id++;
            box_marker.type = visualization_msgs::Marker::CUBE;
            box_marker.action = visualization_msgs::Marker::ADD;
            
            box_marker.pose.position.x = box.center[0];
            box_marker.pose.position.y = box.center[1];
            box_marker.pose.position.z = box.center[2];
            box_marker.pose.orientation.x = box.orientation.x();
            box_marker.pose.orientation.y = box.orientation.y();
            box_marker.pose.orientation.z = box.orientation.z();
            box_marker.pose.orientation.w = box.orientation.w();
            
            box_marker.scale.x = box.dimensions[0];
            box_marker.scale.y = box.dimensions[1];
            box_marker.scale.z = box.dimensions[2];
            
            box_marker.color.r = 0.0;
            box_marker.color.g = 1.0;
            box_marker.color.b = 0.0;
            box_marker.color.a = 0.3;
            
            box_marker.lifetime = ros::Duration(1.0);
            marker_array.markers.push_back(box_marker);
            
            // Text marker (dimensions)
            visualization_msgs::Marker text_marker;
            text_marker.header = header;
            text_marker.ns = "dimensions";
            text_marker.id = id++;
            text_marker.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
            text_marker.action = visualization_msgs::Marker::ADD;
            
            text_marker.pose.position.x = box.center[0];
            text_marker.pose.position.y = box.center[1];
            text_marker.pose.position.z = box.center[2] + box.dimensions[2]/2 + 0.05;
            
            char text[256];
            snprintf(text, sizeof(text), "%.1fx%.1fx%.1fcm", 
                    box.dimensions[0]*100, box.dimensions[1]*100, box.dimensions[2]*100);
            text_marker.text = text;
            
            text_marker.scale.z = 0.02;
            text_marker.color.r = 1.0;
            text_marker.color.g = 1.0;
            text_marker.color.b = 1.0;
            text_marker.color.a = 1.0;
            
            text_marker.lifetime = ros::Duration(1.0);
            marker_array.markers.push_back(text_marker);
            
            // ========== 抓取坐标系可视化（与 generateGrasps 完全一致）==========
            {
                Eigen::Matrix3f R = box.orientation.toRotationMatrix();

                // col(1)=次长轴（夹爪开合），col(2)=最短轴（接近方向，朝向相机）
                Eigen::Vector3f long_axis    = R.col(0);
                Eigen::Vector3f short_axis   = R.col(1);
                Eigen::Vector3f approach_vec = R.col(2);

                // approach_vec 朝向相机（-Z分量>0）
                if (approach_vec.dot(Eigen::Vector3f(0, 0, -1)) < 0)
                    approach_vec = -approach_vec;

                // 此时 box.center 已经位于提取出的表层平面中心
                Eigen::Vector3f top_center = box.center;

                // 夹爪插入方向（朝物体）
                Eigen::Vector3f grasp_z = -approach_vec;

                Eigen::Vector3f grasp_x = short_axis;
                grasp_x -= grasp_x.dot(grasp_z) * grasp_z;
                if (grasp_x.norm() < 1e-6f) {
                    grasp_x = long_axis - long_axis.dot(grasp_z) * grasp_z;
                }
                grasp_x.normalize();

                // 构建夹爪坐标系
                Eigen::Vector3f grasp_y = grasp_z.cross(grasp_x).normalized();
                grasp_x = grasp_y.cross(grasp_z).normalized();
                if (grasp_x.dot(short_axis) < 0.0f) {
                    grasp_x = -grasp_x;
                    grasp_y = -grasp_y;
                }

                // 抓取点位置：顶面向上偏移 offset
                float approach_offset = 0.05f;
                Eigen::Vector3f grasp_origin = top_center + approach_vec * approach_offset;

                // 坐标轴长度
                double grasp_axis_length = 0.08;

                visualization_msgs::Marker grasp_axes_marker;
                grasp_axes_marker.header = header;
                grasp_axes_marker.ns = "grasp_axes";
                grasp_axes_marker.id = id++;
                grasp_axes_marker.type = visualization_msgs::Marker::LINE_LIST;
                grasp_axes_marker.action = visualization_msgs::Marker::ADD;
                grasp_axes_marker.scale.x = 0.008;

                geometry_msgs::Point gp1, gp2;
                gp1.x = grasp_origin[0]; gp1.y = grasp_origin[1]; gp1.z = grasp_origin[2];

                // 夹爪 X 轴（红）= 短边/开合方向
                Eigen::Vector3f gx_end = grasp_origin + grasp_x * grasp_axis_length;
                gp2.x = gx_end[0]; gp2.y = gx_end[1]; gp2.z = gx_end[2];
                grasp_axes_marker.points.push_back(gp1);
                grasp_axes_marker.points.push_back(gp2);
                std_msgs::ColorRGBA c_red; c_red.r=1.0; c_red.a=1.0;
                grasp_axes_marker.colors.push_back(c_red);
                grasp_axes_marker.colors.push_back(c_red);

                // 夹爪 Y 轴（绿）
                Eigen::Vector3f gy_end = grasp_origin + grasp_y * grasp_axis_length;
                gp2.x = gy_end[0]; gp2.y = gy_end[1]; gp2.z = gy_end[2];
                grasp_axes_marker.points.push_back(gp1);
                grasp_axes_marker.points.push_back(gp2);
                std_msgs::ColorRGBA c_green; c_green.g=1.0; c_green.a=1.0;
                grasp_axes_marker.colors.push_back(c_green);
                grasp_axes_marker.colors.push_back(c_green);

                // 夹爪 Z 轴（蓝）= 接近/插入方向（向下）
                Eigen::Vector3f gz_end = grasp_origin + grasp_z * grasp_axis_length;
                gp2.x = gz_end[0]; gp2.y = gz_end[1]; gp2.z = gz_end[2];
                grasp_axes_marker.points.push_back(gp1);
                grasp_axes_marker.points.push_back(gp2);
                std_msgs::ColorRGBA c_blue; c_blue.b=1.0; c_blue.a=1.0;
                grasp_axes_marker.colors.push_back(c_blue);
                grasp_axes_marker.colors.push_back(c_blue);

                grasp_axes_marker.lifetime = ros::Duration(1.0);
                marker_array.markers.push_back(grasp_axes_marker);

                // 抓取点标记（橙色小球）
                visualization_msgs::Marker grasp_point_marker;
                grasp_point_marker.header = header;
                grasp_point_marker.ns = "grasp_points";
                grasp_point_marker.id = id++;
                grasp_point_marker.type = visualization_msgs::Marker::SPHERE;
                grasp_point_marker.action = visualization_msgs::Marker::ADD;
                grasp_point_marker.pose.position.x = grasp_origin[0];
                grasp_point_marker.pose.position.y = grasp_origin[1];
                grasp_point_marker.pose.position.z = grasp_origin[2];
                grasp_point_marker.pose.orientation.w = 1.0;
                grasp_point_marker.scale.x = 0.02;
                grasp_point_marker.scale.y = 0.02;
                grasp_point_marker.scale.z = 0.02;
                grasp_point_marker.color.r = 1.0;
                grasp_point_marker.color.g = 0.5;
                grasp_point_marker.color.b = 0.0;
                grasp_point_marker.color.a = 1.0;
                grasp_point_marker.lifetime = ros::Duration(1.0);
                marker_array.markers.push_back(grasp_point_marker);
            }
        }
        
        marker_pub_.publish(marker_array);
    }
    
    void publishObjectClouds(const std::vector<box_grasp_detection::OrientedBoundingBox>& boxes,
                            const std_msgs::Header& header) {
        // Combine all object clouds
        pcl::PointCloud<pcl::PointXYZ>::Ptr combined_cloud(new pcl::PointCloud<pcl::PointXYZ>);
        
        for (const auto& box : boxes) {
            *combined_cloud += *box.cloud;
        }
        
        sensor_msgs::PointCloud2 cloud_msg;
        pcl::toROSMsg(*combined_cloud, cloud_msg);
        cloud_msg.header = header;
        
        object_cloud_pub_.publish(cloud_msg);
    }
    
    ros::Subscriber cloud_sub_;
    ros::Publisher grasp_pub_;
    ros::Publisher marker_pub_;
    ros::Publisher object_cloud_pub_;

    std::shared_ptr<box_grasp_detection::BoxDetector> detector_;
    
    std::string input_cloud_topic_;
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "box_detector_node");
    
    BoxDetectorNode node;
    
    ros::spin();
    
    return 0;
}
