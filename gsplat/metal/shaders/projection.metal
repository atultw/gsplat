#include <metal_stdlib>
using namespace metal;

// Structure definitions
struct Gaussian {
    float3 mean;        // 3D position
    float4 quat;        // Rotation quaternion (wxyz)
    float3 scale;       // Scale factors
    float opacity;      // Opacity value
};

struct Camera {
    float4x4 viewmat;   // World-to-camera transformation
    float3x3 K;         // Camera intrinsics
    float width;
    float height;
    float near_plane;
    float far_plane;
};

struct Projected2D {
    float2 mean2d;      // 2D projected mean
    float depth;        // Z-depth
    float3 conic;       // Inverse covariance (upper triangle: xx, xy, yy)
    int2 radii;         // Bounding box radii in pixels
    bool valid;         // Whether projection is valid
};

// Convert quaternion and scale to 3D covariance matrix
float3x3 quat_scale_to_covar(float4 q, float3 s) {
    // Normalize quaternion
    q = normalize(q);
    float w = q.x, x = q.y, y = q.z, z = q.w;
    
    // Build rotation matrix from quaternion
    float3x3 R;
    R[0][0] = 1.0 - 2.0 * (y * y + z * z);
    R[0][1] = 2.0 * (x * y - w * z);
    R[0][2] = 2.0 * (x * z + w * y);
    R[1][0] = 2.0 * (x * y + w * z);
    R[1][1] = 1.0 - 2.0 * (x * x + z * z);
    R[1][2] = 2.0 * (y * z - w * x);
    R[2][0] = 2.0 * (x * z - w * y);
    R[2][1] = 2.0 * (y * z + w * x);
    R[2][2] = 1.0 - 2.0 * (x * x + y * y);
    
    // Build scale matrix
    float3x3 S = float3x3(
        float3(s.x, 0.0, 0.0),
        float3(0.0, s.y, 0.0),
        float3(0.0, 0.0, s.z)
    );
    
    // Covariance = R * S * S^T * R^T
    float3x3 RS = R * S;
    return RS * transpose(RS);
}

// Transform Gaussian from world to camera space
void world_to_cam(
    float3 mean_world,
    float3x3 covar_world,
    float4x4 viewmat,
    thread float3& mean_cam,
    thread float3x3& covar_cam
) {
    // Transform mean
    float4 mean_h = float4(mean_world, 1.0);
    float4 mean_cam_h = viewmat * mean_h;
    mean_cam = mean_cam_h.xyz;
    
    // Transform covariance: W = R * Sigma * R^T
    float3x3 R = float3x3(
        viewmat[0].xyz,
        viewmat[1].xyz,
        viewmat[2].xyz
    );
    covar_cam = R * covar_world * transpose(R);
}

// Project 3D Gaussians to 2D
kernel void project_gaussians(
    constant Gaussian* gaussians [[buffer(0)]],
    constant Camera& camera [[buffer(1)]],
    device Projected2D* projected [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    Gaussian g = gaussians[gid];
    
    // Compute 3D covariance from quaternion and scale
    float3x3 covar3d = quat_scale_to_covar(g.quat, exp(g.scale));
    
    // Transform to camera space
    float3 mean_cam;
    float3x3 covar_cam;
    world_to_cam(g.mean, covar3d, camera.viewmat, mean_cam, covar_cam);
    
    // Check if behind camera
    if (mean_cam.z <= camera.near_plane || mean_cam.z >= camera.far_plane) {
        projected[gid].valid = false;
        return;
    }
    
    // Project to 2D using camera intrinsics
    float fx = camera.K[0][0];
    float fy = camera.K[1][1];
    float cx = camera.K[0][2];
    float cy = camera.K[1][2];
    
    float z = mean_cam.z;
    float z2 = z * z;
    
    // 2D projected mean
    float2 mean2d = float2(
        fx * mean_cam.x / z + cx,
        fy * mean_cam.y / z + cy
    );
    
    // Jacobian of perspective projection
    float2x3 J = float2x3(
        float3(fx / z, 0.0, -fx * mean_cam.x / z2),
        float3(0.0, fy / z, -fy * mean_cam.y / z2)
    );
    
    // Project covariance to 2D: Sigma' = J * Sigma * J^T
    float3x2 covar_J = covar_cam * transpose(J);
    float2x2 covar2d = J * covar_J;
    
    // Add a small epsilon for numerical stability
    float eps2d = 0.3;
    covar2d[0][0] += eps2d;
    covar2d[1][1] += eps2d;
    
    // Compute inverse (conic)
    float det = covar2d[0][0] * covar2d[1][1] - covar2d[0][1] * covar2d[1][0];
    if (det <= 0.0) {
        projected[gid].valid = false;
        return;
    }
    
    float inv_det = 1.0 / det;
    float3 conic = float3(
        covar2d[1][1] * inv_det,
        -covar2d[0][1] * inv_det,
        covar2d[0][0] * inv_det
    );
    
    // Compute bounding box radius
    float lambda1 = 0.5 * (covar2d[0][0] + covar2d[1][1] + 
                          sqrt((covar2d[0][0] - covar2d[1][1]) * 
                               (covar2d[0][0] - covar2d[1][1]) + 
                               4.0 * covar2d[0][1] * covar2d[0][1]));
    float radius = ceil(3.0 * sqrt(lambda1));
    
    // Check if visible in image
    if (mean2d.x + radius < 0 || mean2d.x - radius >= camera.width ||
        mean2d.y + radius < 0 || mean2d.y - radius >= camera.height) {
        projected[gid].valid = false;
        return;
    }
    
    // Write output
    projected[gid].mean2d = mean2d;
    projected[gid].depth = z;
    projected[gid].conic = conic;
    projected[gid].radii = int2(int(radius), int(radius));
    projected[gid].valid = true;
}
