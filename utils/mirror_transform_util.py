import torch
from scene import Camera
def skew_sym_mat(x):
    device = x.device
    dtype = x.dtype
    ssm = torch.zeros(3, 3, device=device, dtype=dtype)
    ssm[0, 1] = -x[2]
    ssm[0, 2] = x[1]
    ssm[1, 0] = x[2]
    ssm[1, 2] = -x[0]
    ssm[2, 0] = -x[1]
    ssm[2, 1] = x[0]
    return ssm

def V(theta):
    dtype = theta.dtype
    device = theta.device
    I = torch.eye(3, device=device, dtype=dtype)
    W = skew_sym_mat(theta)
    W2 = W @ W
    angle = torch.norm(theta)
    if angle < 1e-5:
        V = I + 0.5 * W + (1.0 / 6.0) * W2
    else:
        V = (
            I
            + W * ((1.0 - torch.cos(angle)) / (angle**2))
            + W2 * ((angle - torch.sin(angle)) / (angle**3))
        )
    return V
def SO3_exp(theta):
    device = theta.device
    dtype = theta.dtype

    W = skew_sym_mat(theta)
    W2 = W @ W
    angle = torch.norm(theta)
    I = torch.eye(3, device=device, dtype=dtype)
    if angle < 1e-5:
        return I + W + 0.5 * W2
    else:
        return (
            I
            + (torch.sin(angle) / angle) * W
            + ((1 - torch.cos(angle)) / (angle**2)) * W2
        )

def SE3_exp(tau):
    dtype = tau.dtype
    device = tau.device

    rho = tau[:3]
    theta = tau[3:]
    R = SO3_exp(theta)
    t = V(theta) @ rho

    T = torch.eye(4, device=device, dtype=dtype)
    T[:3, :3] = R
    T[:3, 3] = t
    return T

def update_pose(camera : Camera, converged_threshold=1e-4):
    tau = torch.cat([camera.cam_trans_delta_mirror, camera.cam_rot_delta_mirror], axis=0)

    T_w2c = torch.eye(4, device=tau.device)

    view_before = camera.world_view_transform_mirror
    R = view_before[:3,:3]
    T = view_before[3,:3]
    # T_w2c[0:3, 0:3] = torch.from_numpy( camera.R)
    # T_w2c[0:3, 3] = torch.from_numpy(camera.T)

    T_w2c[0:3, 0:3] = R
    T_w2c[0:3, 3] = T

    new_w2c = SE3_exp(tau) @ T_w2c

    new_R = new_w2c[0:3, 0:3]
    new_T = new_w2c[0:3, 3]

    converged = tau.norm() < converged_threshold

    new_world_view_transform = getWorld2View2(new_R, new_T)

    # camera.cam_rot_delta.data.fill_(0)
    # camera.cam_trans_delta.data.fill_(0)
    # camera.cam_rot_delta*=0
    # camera.cam_trans_delta*=0

    return converged , new_world_view_transform



def getWorld2View2(R, t, translate=torch.tensor([0.0, 0.0, 0.0]), scale=1.0):
    translate = translate.to(R.device)
    Rt = torch.zeros((4, 4), device=R.device)
    # Rt[:3, :3] = R.transpose()
    Rt[:3, :3] = R
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = torch.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    RT = torch.linalg.inv(C2W)
    return RT.transpose(0, 1)


def calculate_loss(camera:Camera,mirror_transform):
    with torch.no_grad():
        converged,viewpoint_after = update_pose(camera)
    # viewpoint_before = camera.world_view_transform_mirror

    return viewpoint_after


    # w2c = viewmatrix.transpose(0, 1)  # Q_o
    #     viewmatrix = torch.matmul(w2c, mirror_transform.inverse()).transpose(0, 1)
    #     projmatrix = (viewmatrix.unsqueeze(0).bmm(viewpoint_camera.projection_matrix.unsqueeze(0))).squeeze(0)
    #     campos = viewmatrix.inverse()[3, :3]

