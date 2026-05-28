import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import zipfile
import tempfile
import pydicom
import numpy as np
import torch
import torch.nn as nn
import trimesh
import open3d as o3d
import torch.nn.functional as F
import matplotlib.pyplot as plt
from scipy.stats import zscore
import scipy.ndimage
from skimage import measure
from torch.utils.data import Dataset
st.set_page_config(layout="wide")


N_IN_POINTS = 2674

#uploaded_file = st.file_uploader("DICOM zip 파일을 업로드하세요.", type="zip")
#save_dir = st.text_input("저장할 디렉토리 경로 (예: D:/데이터 증강/zip_features)", value="D:/데이터 증강/zip_features")
#os.makedirs(save_dir, exist_ok=True)

#uploaded_name = uploaded_file.name
#BASE_NAME = os.path.splitext(uploaded_name)[0]  # 예시 (혹은 파일명에서 자동 추출 가능)



def find_dicom_files(root):
    dcm_files = []
    for path, _, files in os.walk(root):
        for f in files:
            if f.lower().endswith('.dcm'):
                dcm_files.append(os.path.join(path, f))
    return sorted(dcm_files)

def load_scan(input_folder):
    dcm_files = find_dicom_files(input_folder)
    slices = [pydicom.dcmread(f) for f in dcm_files]
    try:
        slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
    except Exception:
        slices.sort(key=lambda x: float(x.SliceLocation))
    return slices

def get_pixels_hu(slices):
    image = np.stack([s.pixel_array for s in slices]).astype(np.int16)
    for i, s in enumerate(slices):
        intercept, slope = s.RescaleIntercept, s.RescaleSlope
        if slope != 1:
            image[i] = (image[i].astype(np.float64) * slope).astype(np.int16)
        image[i] += np.int16(intercept)
    return image

def resample(image, scan, new_spacing=[1,1,1]):
    spacing = np.array(
        [float(scan[0].SliceThickness)] + list(map(float, scan[0].PixelSpacing)),
        dtype=np.float32
    )
    resize_factor = spacing / new_spacing
    new_shape     = np.round(image.shape * resize_factor)
    real_factor   = new_shape / image.shape
    return scipy.ndimage.zoom(image, real_factor, mode='nearest'), spacing

def segment_lung_mask(image, fill_lung_structures=True):
    binary = (image > -400).astype(np.int8) + 1
    labels = measure.label(binary)
    background = labels[0,0,0]
    binary[labels==background] = 2
    if fill_lung_structures:
        for i, sl in enumerate(binary):
            lab = measure.label(sl-1)
            lm  = np.argmax(np.bincount(lab.flat)[1:])+1 if lab.max()>0 else None
            if lm is not None:
                sl[lab!=lm] = 1
            binary[i] = sl + 1
    binary -= 1
    binary = 1 - binary
    labels = measure.label(binary, background=0)
    lm = np.argmax(np.bincount(labels.flat)[1:])+1 if labels.max()>0 else None
    if lm is not None:
        binary[labels!=lm] = 0
    return binary

def build_features_from_verts(verts):
    # verts: (N, 3), np.float32, 반드시 (x, y, z) 순서
    center = verts.mean(axis=0)
    dist_center = np.linalg.norm(verts - center, axis=1, keepdims=True)
    dist_center_norm = (dist_center - dist_center.min()) / (dist_center.max() - dist_center.min() + 1e-8)
    verts_z = zscore(verts, axis=0)
    features_7 = np.concatenate([verts, dist_center_norm, verts_z], axis=1)
    return features_7

#------------MeshCNN 전처리-----------------
def mesh_load_scan(input_folder):
    dcm_files = sorted(glob.glob(os.path.join(input_folder, "**", "*.dcm"), recursive=True))
    if len(dcm_files) < 2:
        raise FileNotFoundError("❌ 유효한 DICOM 파일이 2개 이상 필요합니다.")
    slices = [pydicom.dcmread(f) for f in dcm_files]
    slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
    try:
        slice_thickness = abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = abs(slices[0].SliceLocation - slices[1].SliceLocation)
    for s in slices:
        s.SliceThickness = slice_thickness
    return slices

def mesh_get_pixels_hu(slices):
    image = np.stack([s.pixel_array for s in slices]).astype(np.int16)
    image[image == -2000] = 0
    for i, s in enumerate(slices):
        intercept = s.RescaleIntercept
        slope = s.RescaleSlope
        if slope != 1:
            image[i] = (image[i].astype(np.float64) * slope).astype(np.int16)
        image[i] += np.int16(intercept)
    return image

def mesh_resample(image, scan, new_spacing=[1, 1, 1]):
    spacing = np.array([
        float(scan[0].SliceThickness),
        float(scan[0].PixelSpacing[0]),
        float(scan[0].PixelSpacing[1])
    ])
    resize_factor = spacing / new_spacing
    new_shape = np.round(image.shape * resize_factor)
    real_factor = new_shape / image.shape
    image_resampled = scipy.ndimage.zoom(image, real_factor, mode='nearest')
    return image_resampled, spacing

def mesh_segment_lung_mask(image, threshold=-600):
    binary = (image > threshold).astype(np.int8) + 1
    labels = measure.label(binary)
    background = labels[0, 0, 0]
    binary[labels == background] = 2
    binary -= 1
    binary = 1 - binary
    labels = measure.label(binary, background=0)
    if labels.max() > 0:
        largest = labels == np.argmax(np.bincount(labels.flat)[1:]) + 1
        binary[~largest] = 0
    return binary

def mesh_lung_mask_to_mesh(segmented, min_edges=500):
    verts, faces, _, _ = measure.marching_cubes(segmented, level=0.5)
    approx_edges = len(faces) * 3 // 2
    if approx_edges < min_edges:
        raise ValueError(f"❌ 메시 에지 수 부춸: {approx_edges}개")
    return trimesh.Trimesh(vertices=verts, faces=faces)

def mesh_visualize_segmentation(mesh, edge_labels):
    face_colors = np.tile([200, 200, 200], (len(mesh.faces), 1))
    edge_face_map = {}
    edge_keys = []
    for f_idx, face in enumerate(mesh.faces):
        for e in [(face[0], face[1]), (face[1], face[2]), (face[2], face[0])]:
            e = tuple(sorted(e))
            if e not in edge_face_map:
                edge_keys.append(e)
            edge_face_map.setdefault(e, []).append(f_idx)
    highlight_faces = set()
    for i, lbl in enumerate(edge_labels):
        if lbl == 1 and i < len(edge_keys):
            highlight_faces.update(edge_face_map.get(edge_keys[i], []))
    face_colors[list(highlight_faces)] = [255, 0, 0]
    mesh.visual.face_colors = face_colors
    return mesh

@st.fragment
def show_ct_viewer(ct_volume):
    st.subheader("🩻 CT 슬라이스 뷰어")

    # 슬라이더 인덱스를 고유 키로 지정해서 Streamlit의 상태로 관리
    if "slice_index" not in st.session_state:
        st.session_state.slice_index = ct_volume.shape[0] // 2  # 초기값 설정

    # 슬라이더 UI만 업데이트하고 값은 세션 상태에 저장
    st.session_state.slice_index = st.slider(
        "Z축 슬라이스 인덱스",
        0,
        ct_volume.shape[0] - 1,
        value=st.session_state.slice_index,
        key="slice_slider"
    )

    # 슬라이서에서 선택된 인덱스를 기반으로 이미지를 렌더링
    idx = st.session_state.slice_index
    slice_img = ct_volume[idx]
    slice_img = np.clip(slice_img, -1000, 400)
    norm_img = ((slice_img + 1000) / 1400 * 255).astype(np.uint8)

    # 이미지만 갱신
    st.image(norm_img, caption=f"Axial CT Slice #{idx}", width=600)

#-------------MeshCNN 유틸 함수--------------
def trimesh_to_open3d(mesh: trimesh.Trimesh):
    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(mesh.vertices)
    o3d_mesh.triangles = o3d.utility.Vector3iVector(mesh.faces)
    return o3d_mesh

def open3d_to_trimesh(o3d_mesh: o3d.geometry.TriangleMesh):
    return trimesh.Trimesh(
        vertices=np.asarray(o3d_mesh.vertices),
        faces=np.asarray(o3d_mesh.triangles),
        process=False
    )

def simplify_mesh(mesh: trimesh.Trimesh, target_faces=60000):
    o3d_mesh = trimesh_to_open3d(mesh)
    simplified = o3d_mesh.simplify_quadric_decimation(target_faces)
    return open3d_to_trimesh(simplified)

def preprocess_mesh_single(mesh: trimesh.Trimesh, max_edges=6000000, normalize=True,
                           max_faces=200000, target_faces=60000):
    """
    메쉬를 전처리합니다. 너무 크면 자동으로 단순화합니다.
    """
    if len(mesh.faces) > max_faces:
        mesh = simplify_mesh(mesh, target_faces)

    # 이하 기존 전처리 로직 유지...
    vertices = np.array(mesh.vertices, dtype=np.float32)
    faces = np.array(mesh.faces, dtype=np.int32)

    if normalize:
        c = vertices.mean(axis=0, keepdims=True)
        vertices -= c
        scale = np.max(np.linalg.norm(vertices, axis=1, keepdims=True))
        vertices /= (scale + 1e-8)

    v1 = vertices[faces[:, 0]]
    v2 = vertices[faces[:, 1]]
    v3 = vertices[faces[:, 2]]
    normals = np.cross(v2 - v1, v3 - v1)
    normals = normals / (np.linalg.norm(normals, axis=1, keepdims=True) + 1e-8)

    edge_to_faces = {}
    for i, tri in enumerate(faces):
        for e in [(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])]:
            e = tuple(sorted(e))
            edge_to_faces.setdefault(e, []).append(i)
    edges = list(edge_to_faces.keys())

    if len(edges) == 0 or len(edges) > max_edges:
        raise ValueError(f"Too many or too few edges: {len(edges)}")

    vert_to_edges = {v: [] for v in range(vertices.shape[0])}
    for i, (v1, v2) in enumerate(edges):
        vert_to_edges[v1].append(i)
        vert_to_edges[v2].append(i)

    neigh0 = []
    for i, (v1, v2) in enumerate(edges):
        nbs = set(vert_to_edges[v1] + vert_to_edges[v2])
        nbs.discard(i)
        lst = list(nbs)[:4]
        while len(lst) < 4:
            lst.append(i)
        neigh0.append(lst)
    neigh0 = np.array(neigh0, dtype=np.int32)

    def edge_feature(i, e):
        p1, p2 = vertices[e[0]], vertices[e[1]]
        edge_len = np.linalg.norm(p1 - p2) + 1e-8
        faces_idx = edge_to_faces[e]

        def comp(feat_idx):
            tri = faces[feat_idx]
            opp_v = [v for v in tri if v not in e][0]
            vec1 = vertices[e[0]] - vertices[opp_v]
            vec2 = vertices[e[1]] - vertices[opp_v]
            inner = np.arccos(np.clip(np.dot(vec1, vec2) /
                       (np.linalg.norm(vec1) * np.linalg.norm(vec2) + 1e-8), -1, 1))
            area = np.linalg.norm(np.cross(vec1, vec2)) / 2.0
            h = (2 * area) / (edge_len + 1e-8)
            ratio = edge_len / (h + 1e-8)
            return inner, ratio

        if len(faces_idx) >= 1:
            i1, r1 = comp(faces_idx[0])
        else:
            i1, r1 = 0.0, 0.0
        if len(faces_idx) == 2:
            i2, r2 = comp(faces_idx[1])
            n1, n2 = normals[faces_idx[0]], normals[faces_idx[1]]
            cos_theta = np.clip(np.dot(n1, n2), -1., 1.)
            dihedral = np.arccos(cos_theta)
        else:
            i2, r2 = i1, r1
            dihedral = 0.0

        return [dihedral, i1, i2, r1, r2]

    edge_feats0 = np.array([edge_feature(i, e) for i, e in enumerate(edges)], dtype=np.float32)

    def create_pool(cmap_size):
        if cmap_size % 2 == 0:
            E1 = cmap_size // 2
        else:
            E1 = cmap_size // 2 + 1
        cmap = np.empty(cmap_size, dtype=np.int32)
        for i in range(cmap_size // 2):
            cmap[2*i] = i
            cmap[2*i+1] = i
        if cmap_size % 2 == 1:
            cmap[-1] = E1 - 1
        return cmap, E1

    cmap1, E1 = create_pool(len(edges))
    neigh1 = [[] for _ in range(E1)]
    for i_cl in range(E1):
        members = np.where(cmap1 == i_cl)[0]
        nbs = set()
        for m in members:
            for n in neigh0[m]:
                j_cl = cmap1[n]
                if j_cl != i_cl:
                    nbs.add(j_cl)
        lst = list(nbs)[:4]
        while len(lst) < 4:
            lst.append(i_cl)
        neigh1[i_cl] = lst
    neigh1 = np.array(neigh1, dtype=np.int32)

    cmap2, E2 = create_pool(E1)
    neigh2 = [[] for _ in range(E2)]
    for i_cl in range(E2):
        members = np.where(cmap2 == i_cl)[0]
        nbs = set()
        for m in members:
            for n in neigh1[m]:
                j_cl = cmap2[n]
                if j_cl != i_cl:
                    nbs.add(j_cl)
        lst = list(nbs)[:4]
        while len(lst) < 4:
            lst.append(i_cl)
        neigh2[i_cl] = lst
    neigh2 = np.array(neigh2, dtype=np.int32)

    return {
        'edge_feats0': edge_feats0,
        'neigh0': neigh0,
        'neigh1': neigh1,
        'neigh2': neigh2,
        'cmap1': cmap1,
        'cmap2': cmap2
    }

def load_npz_for_inference(npz_path: str):
    data = np.load(npz_path)
    return {
        'edge_feats0': torch.from_numpy(data['edge_feats0']).float(),
        'neigh0': torch.from_numpy(data['neigh0']).long(),
        'neigh1': torch.from_numpy(data['neigh1']).long(),
        'neigh2': torch.from_numpy(data['neigh2']).long(),
        'cmap1': torch.from_numpy(data['cmap1']).long(),
        'cmap2': torch.from_numpy(data['cmap2']).long(),
    }

#------------- MeshCNN 모델 정의---------------
class MeshConv(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.linear = nn.Linear(in_c * 5, out_c)
    def forward(self, x, neigh):
        x_nbr = x[neigh]
        x_cat = torch.cat([x.unsqueeze(1), x_nbr], dim=1).reshape(x.shape[0], -1)
        return self.linear(x_cat)

class MeshPool(nn.Module):
    def forward(self, x_prev, cmap):
        E_prev, C = x_prev.shape
        E_next = int(cmap.max().item()) + 1
        x_next = x_prev.new_zeros((E_next, C))
        cnt = x_prev.new_zeros((E_next, C))
        x_next.index_add_(0, cmap, x_prev)
        cnt.index_add_(0, cmap, x_prev.new_ones((E_prev, C)))
        return x_next / (cnt + 1e-8)

class MeshCNN(nn.Module):
    def __init__(self, in_c=5):
        super().__init__()
        self.conv0 = MeshConv(in_c, 16)
        self.pool1 = MeshPool()
        self.conv1 = MeshConv(16, 32)
        self.pool2 = MeshPool()
        self.conv2 = MeshConv(32, 64)
        self.fc1 = nn.Linear(64, 100)
        self.fc2 = nn.Linear(100, 2)
        self.seg_head = nn.Linear(16, 2)
    def forward(self, edge_feats0, neigh0, neigh1, neigh2, cmap1, cmap2, mode='both'):
        x0 = F.relu(self.conv0(edge_feats0, neigh0))
        x1 = self.pool1(x0, cmap1)
        x1 = F.relu(self.conv1(x1, neigh1))
        x2 = self.pool2(x1, cmap2)
        x2 = F.relu(self.conv2(x2, neigh2))
        xg = torch.max(x2, dim=0)[0]
        cls_logits = F.relu(self.fc1(xg))
        cls_logits = self.fc2(cls_logits)
        seg_logits = self.seg_head(x0)
        if mode == 'classification':
            return cls_logits.unsqueeze(0)
        elif mode == 'segmentation':
            return seg_logits
        else:
            return cls_logits.unsqueeze(0), seg_logits


# -------------- 모델 정의 ------------------
class ImbalancedCTNoduleDataset(Dataset):
    def __init__(self, verts_dir, labels_dir, bases, augment=False):
        self.verts_dir = verts_dir
        self.labels_dir = labels_dir
        self.bases = bases
        self.augment = augment

        # 각 샘플의 positive 비율 계산 (가중치 샘플링용)
        self.sample_weights = []
        for base in bases:
            labels = np.load(os.path.join(labels_dir, base + '_vertex_labels.npy'))
            pos_ratio = labels.mean()
            # positive가 많은 샘플에 더 높은 가중치
            weight = min(pos_ratio * 10 + 0.1, 2.0)  # 최대 2배까지 가중치
            self.sample_weights.append(weight)

    def __len__(self):
        return len(self.bases)

    def get_sample_weights(self):
        return self.sample_weights

    def __getitem__(self, i):
        base = self.bases[i]
        verts = np.load(os.path.join(self.verts_dir, base + '_features.npy'))
        labels = np.load(os.path.join(self.labels_dir, base + '_vertex_labels.npy'))

        # 간단한 augmentation (결절이 있는 경우)
        if self.augment and labels.mean() > 0.01:
            # 작은 노이즈 추가
            noise = np.random.normal(0, 0.001, verts.shape)
            verts = verts + noise

        return torch.tensor(verts, dtype=torch.float32), torch.tensor(labels, dtype=torch.float32), base

class SimplePointTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads=4, ff_hidden=256, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, ff_hidden),
            nn.ReLU(),
            nn.Linear(ff_hidden, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        # x: (B, V, C)
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h)
        x = x + attn_out
        h2 = self.norm2(x)
        x = x + self.ff(h2)
        return x

class HybridSpiralNetPointNetTransformer(nn.Module):
    def __init__(self, spiralnet, in_channels, hidden_dim=256, out_dim=1, dropout=0.3,
                 transformer_heads=4, transformer_blocks=1):
        super().__init__()
        self.spiralnet = spiralnet  # 기존 SpiralNet 인스턴스

        # PointNet branch
        self.pointnet_mlp1 = nn.Linear(in_channels, hidden_dim)
        self.pointnet_bn1 = nn.BatchNorm1d(hidden_dim)
        self.pointnet_mlp2 = nn.Linear(hidden_dim, hidden_dim)
        self.pointnet_bn2 = nn.BatchNorm1d(hidden_dim)
        self.pointnet_dropout = nn.Dropout(dropout)

        # Transformer branch (Point Attention)
        self.transformer_blocks = nn.ModuleList([
            SimplePointTransformerBlock(hidden_dim, num_heads=transformer_heads, dropout=dropout)
            for _ in range(transformer_blocks)
        ])

        # 최종적으로 vertex별로 결합 (B, V, hidden*2)
        self.pointnet_head = nn.Linear(hidden_dim * 2, out_dim)

    def forward(self, x, spiral_idx):
        # x: (B, V, C)
        spiral_out = self.spiralnet(x, spiral_idx)  # (B, V)

        B, V, C = x.size()
        # ----- PointNet branch -----
        # Local MLP1
        feat1 = self.pointnet_mlp1(x)  # (B, V, hidden_dim)
        feat1 = feat1.view(B * V, -1)
        feat1 = self.pointnet_bn1(feat1)
        feat1 = F.relu(feat1)
        feat1 = feat1.view(B, V, -1)

        # Local MLP2
        feat2 = self.pointnet_mlp2(feat1)
        feat2 = feat2.view(B * V, -1)
        feat2 = self.pointnet_bn2(feat2)
        feat2 = F.relu(feat2)
        feat2 = feat2.view(B, V, -1)

        point_feat = self.pointnet_dropout(feat2)

        # --- Transformer Attention 추가 ---
        for block in self.transformer_blocks:
            point_feat = block(point_feat)  # (B, V, hidden_dim)

        # Global pooling
        global_feat, _ = torch.max(point_feat, dim=1, keepdim=True)  # (B, 1, hidden_dim)
        global_feat = global_feat.expand(-1, V, -1)  # (B, V, hidden_dim)

        # concat local+global features
        pn_feat = torch.cat([point_feat, global_feat], dim=2)  # (B, V, hidden_dim*2)
        pointnet_out = self.pointnet_head(pn_feat).squeeze(-1)  # (B, V)

        # Soft ensemble (SpiralNet + PointNet)
        out = (spiral_out + pointnet_out) / 2
        return out

# SpiralNet + PointNet 하이브리드 모델 클래스
class HybridSpiralNetPointNet(nn.Module):
    def __init__(self, spiralnet, in_channels, hidden_dim=256, out_dim=1, dropout=0.3):
        super().__init__()
        self.spiralnet = spiralnet  # 기존 SpiralNet 인스턴스

        # PointNet branch: (B, V, C) → (B, V, hidden) → (B, hidden)
        self.pointnet_mlp1 = nn.Linear(in_channels, hidden_dim)
        self.pointnet_bn1 = nn.BatchNorm1d(hidden_dim)
        self.pointnet_mlp2 = nn.Linear(hidden_dim, hidden_dim)
        self.pointnet_bn2 = nn.BatchNorm1d(hidden_dim)
        self.pointnet_dropout = nn.Dropout(dropout)
        # 최종적으로 vertex별로 결합 (B, V, hidden*2)
        self.pointnet_head = nn.Linear(hidden_dim * 2, out_dim)

    def forward(self, x, spiral_idx):
        # x: (B, V, C)
        spiral_out = self.spiralnet(x, spiral_idx)  # (B, V)

        B, V, C = x.size()
        # ----- PointNet branch -----
        # Local MLP1
        feat1 = self.pointnet_mlp1(x)  # (B, V, hidden_dim)
        feat1 = feat1.view(B * V, -1)  # (B*V, hidden_dim)
        feat1 = self.pointnet_bn1(feat1)
        feat1 = F.relu(feat1)
        feat1 = feat1.view(B, V, -1)  # (B, V, hidden_dim)

        # Local MLP2
        feat2 = self.pointnet_mlp2(feat1)
        feat2 = feat2.view(B * V, -1)
        feat2 = self.pointnet_bn2(feat2)
        feat2 = F.relu(feat2)
        feat2 = feat2.view(B, V, -1)  # (B, V, hidden_dim)

        point_feat = self.pointnet_dropout(feat2)  # (B, V, hidden_dim)

        # Global pooling
        global_feat, _ = torch.max(point_feat, dim=1, keepdim=True)  # (B, 1, hidden_dim)
        global_feat = global_feat.expand(-1, V, -1)  # (B, V, hidden_dim)

        # concat local+global features
        pn_feat = torch.cat([point_feat, global_feat], dim=2)  # (B, V, hidden_dim*2)
        pointnet_out = self.pointnet_head(pn_feat).squeeze(-1)  # (B, V)

        # Soft ensemble (SpiralNet + PointNet)
        out = (spiral_out + pointnet_out) / 2
        return out

class HybridSpiralNetMLP(nn.Module):
    def __init__(self, spiralnet, mlp_hidden=256, num_classes=1):
        super().__init__()
        self.spiralnet = spiralnet  # 기존 SpiralNet 구조
        in_dim = spiralnet.in_channels
        # MLP branch (예: 3-layer)
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, mlp_hidden),
            nn.ReLU(),
            nn.Linear(mlp_hidden, mlp_hidden//2),
            nn.ReLU(),
            nn.Linear(mlp_hidden//2, num_classes)
        )
        # 최종 head (concat된 feature로)
        self.head = nn.Sequential(
            nn.Linear(2 * num_classes, num_classes)  # spiral + mlp output concat
        )
    def forward(self, x, spiral_idx):
        # x: (B, V, C)
        # SpiralNet output (B, V)
        out_spiral = self.spiralnet(x, spiral_idx)
        # MLP output (B, V)
        out_mlp = self.mlp(x)
        # concat outputs
        out = torch.cat([out_spiral.unsqueeze(-1), out_mlp], dim=-1)
        out = self.head(out).squeeze(-1)
        return out

# ────────────────────────────────────────────────────
# SpiralNet Model (기존과 동일)
# ────────────────────────────────────────────────────
class SpiralConv(nn.Module):
    def __init__(self, in_channels, out_channels, spiral_len: int):
        super().__init__()
        self.spiral_len = spiral_len
        self.conv = nn.Conv1d(in_channels * spiral_len, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, spiral_idx):
        B, C, V = x.size()
        L = spiral_idx.size(1)
        idx = spiral_idx.unsqueeze(0).unsqueeze(0).expand(B, C, V, L)
        x_expand = x.unsqueeze(-1).expand(B, C, V, L)
        neigh = torch.gather(x_expand, 2, idx)
        neigh = neigh.reshape(B, C * L, V)
        out = self.conv(neigh)
        out = self.bn(out)
        return self.relu(out)

class SpiralBlock(nn.Module):
    def __init__(self, in_channels, out_channels, spiral_index, dropout=0.0):
        super().__init__()
        spiral_len = spiral_index.shape[1]
        self.spiral = SpiralConv(in_channels, out_channels, spiral_len)
        self.use_res = (in_channels == out_channels)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x, spiral_idx):
        out = self.spiral(x, spiral_idx)
        out = self.dropout(out)
        if self.use_res:
            return out + x
        else:
            return out

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        w = self.pool(x).squeeze(-1)
        w = self.fc(w).unsqueeze(-1)
        return x * w

class ImprovedSpiralNet(nn.Module):
    """개선된 SpiralNet with Multi-scale features"""
    def __init__(self, in_channels, hidden_channels, out_channels,
                 num_blocks, spiral_indices=None, dropout=0.0, use_se=True):
        super().__init__()
        self.spiral_indices = spiral_indices
        num_blocks = len(spiral_indices)
        self.use_se = use_se

        self.encoder = nn.ModuleList()
        for i in range(num_blocks):
            in_c = in_channels if i == 0 else hidden_channels
            out_c = hidden_channels
            spiral_index = spiral_indices[i]
            self.encoder.append(
                SpiralBlock(in_c, out_c, spiral_index, dropout=dropout)
            )

        if use_se:
            self.se_blocks = nn.ModuleList([
                SEBlock(hidden_channels) for _ in range(num_blocks)
            ])
        else:
            self.se_blocks = [nn.Identity()] * num_blocks

        self.skip_conns = nn.ModuleList()
        self.decoder = nn.ModuleList()
        for i in range(num_blocks - 1):
            self.skip_conns.append(nn.Conv1d(hidden_channels * 2, hidden_channels, kernel_size=1))
            spiral_index_dec = spiral_indices[num_blocks - 2 - i]
            self.decoder.append(
                SpiralBlock(hidden_channels, hidden_channels, spiral_index_dec, dropout=dropout)
            )

        # Multi-scale classifier
        self.classifier = nn.Sequential(
            nn.Conv1d(hidden_channels, hidden_channels // 2, kernel_size=1),
            nn.BatchNorm1d(hidden_channels // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_channels // 2, hidden_channels // 4, kernel_size=1),
            nn.BatchNorm1d(hidden_channels // 4),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_channels // 4, out_channels, kernel_size=1)
        )

    def forward(self, verts, indices):
        x = verts.permute(0, 2, 1)
        features = []

        for block, idx, se in zip(self.encoder, self.spiral_indices, self.se_blocks):
            x = block(x, idx)
            x = se(x)
            features.append(x)

        for skip_conv, dec_block, idx, feat in zip(
            self.skip_conns,
            self.decoder,
            reversed(self.spiral_indices[:-1]),
            reversed(features[:-1])
        ):
            x = torch.cat([x, feat], dim=1)
            x = skip_conv(x)
            x = dec_block(x, idx)

        out = self.classifier(x)
        return out.squeeze(1)


#------------ 모델 준비 --------------
# 예시 spiral_idx 불러오기
SPIRAL_PATH = "C:/Users/<PC_A>/Desktop/spiral_torch/spiral_9.npy"
spiral_all = np.load(SPIRAL_PATH, allow_pickle=True)  # (V, K)
num_blocks = 4
feature_dim = 7
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
spiral_idx = [torch.from_numpy(spiral_all).long().to(DEVICE) for _ in range(num_blocks)]


spiralnet = ImprovedSpiralNet(
    in_channels=feature_dim,
    hidden_channels=256,
    out_channels=1,
    num_blocks=num_blocks,
    spiral_indices=spiral_idx,
    dropout=0.3,
    use_se=True
).to(DEVICE)

model = HybridSpiralNetPointNetTransformer(
    spiralnet=spiralnet,
    in_channels=feature_dim,
    hidden_dim=256,
    out_dim=1,
    dropout=0.3,
    transformer_heads=4,
    transformer_blocks=1
).to(DEVICE)

WEIGHT_PATH = "C:/Users/<PC_A>/Desktop/spiral_torch/dhkstjd.pth"
ckpt = torch.load(WEIGHT_PATH, map_location=DEVICE)
model.load_state_dict(ckpt['model_state_dict'], strict=True)
#best_threshold = ckpt.get('threshold', 0.5)
best_threshold = 0.5
model.eval()

# MeshCNN 로드
MODEL_PATH = "C:/Users/<PC_A>/Desktop/spiral_torch/classification_v.1.0.pth"   # 예시 경로
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

meshcnn_model = MeshCNN()
meshcnn_model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
meshcnn_model.to(DEVICE)
meshcnn_model.eval()


# 메인


st.markdown("""
    <style>
    .st-emotion-cache-13k62yr {
        padding-top: 2.5rem;  /* 상단 여백 */
    }
    .viewer-box {
        background: #eaf6fe;
        border-radius: 14px;
        padding: 10px 10px;
        min-height: 40px;
        font-size: 15px;
        margin-bottom: 15px;
    }

    .stDownloadButton {
        margin-top: 18px;
    }
    .left-box {
        background: #e9fcff;
        border-radius: 10px;
        padding: 10px 10px 5px 10px;
        margin-bottom: 10px;
        min-height: 60px;
    }
    .center-announce {
        text-align: center;
        margin-top: 40px;
        color: #888;
        font-size: 13px;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h2 style='text-align:center; margin-bottom:1.2rem;'>🫁AI 기반 폐 CT 3D 시각화 및 결절 탐지 시스템</h2>", unsafe_allow_html=True)
left, right = st.columns([1, 1])
with left:
    #st.markdown("<div class='left-box'><b>DICOM 파일을 업로드하세요.</b></div>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("DICOM zip 파일을 업로드하세요.", type="zip")
    if uploaded_file is not None:
        st.success("업로드 완료! 압축 해제 중...")
        filename = uploaded_file.name
        BASE_NAME = os.path.splitext(filename)[0]  # .zip 제거
        TRAIN_VERT_PATH = f'D:/데이터 증강/결절_2674/{BASE_NAME}_s100_verts.npy'
        TRAIN_LABEL_PATH = f'D:/데이터 증강/라벨_2674/{BASE_NAME}_s100_vertex_labels.npy'

        with tempfile.TemporaryDirectory() as tmpdir:
            # ZIP 저장 및 해제
            zippath = os.path.join(tmpdir, "uploaded.zip")
            with open(zippath, "wb") as f:
                f.write(uploaded_file.read())
            with zipfile.ZipFile(zippath, 'r') as zip_ref:
                zip_ref.extractall(tmpdir)

            dicom_files = find_dicom_files(tmpdir)
            slices = [pydicom.dcmread(f) for f in dicom_files]
            image = get_pixels_hu(slices)  # (Z, Y, X)

            
            if "preprocessed" not in st.session_state:
                st.session_state["preprocessed"] = {}  # 빈 딕셔너리로 초기화

            st.session_state["preprocessed"]["image"] = image

            # MeshCNN 분류 예측
            mesh_files = [f for f in os.listdir(tmpdir) if f.endswith('.obj') or f.endswith('.ply')]
            if len(mesh_files) == 0:
                image = get_pixels_hu(slices)
                resampled, _ = resample(image, slices, new_spacing=[1,1,1])
                segmented = segment_lung_mask(resampled, fill_lung_structures=False)
                verts, faces, _, _ = measure.marching_cubes(segmented, level=0.5)
                mesh = trimesh.Trimesh(vertices=verts, faces=faces)
                mesh_path = os.path.join(tmpdir, "generated_mesh.ply")
                mesh.export(mesh_path)
            else:
                mesh_path = os.path.join(tmpdir, mesh_files[0])

            # MeshCNN 추론
            try:
                mesh = trimesh.load(mesh_path, process=False)
                meshcnn_input = preprocess_mesh_single(mesh)
                index_keys = ['neigh0', 'neigh1', 'neigh2', 'cmap1', 'cmap2']
                for k in meshcnn_input:
                    arr = meshcnn_input[k]
                    if isinstance(arr, np.ndarray):
                        if k in index_keys:
                            meshcnn_input[k] = torch.from_numpy(arr).long().to(DEVICE)
                        else:
                            meshcnn_input[k] = torch.from_numpy(arr).float().to(DEVICE)
                    else:
                        meshcnn_input[k] = arr.to(DEVICE)
                with torch.no_grad():
                    cls_logits = meshcnn_model(
                        meshcnn_input['edge_feats0'],
                        meshcnn_input['neigh0'],
                        meshcnn_input['neigh1'],
                        meshcnn_input['neigh2'],
                        meshcnn_input['cmap1'],
                        meshcnn_input['cmap2'],
                        mode='classification'
                    )
                    probs = torch.softmax(cls_logits, dim=1).cpu().numpy().squeeze()
                    pred_label = int(np.argmax(probs))
                    negative_pct = probs[0] * 100
                    positive_pct = probs[1] * 100

                # -- 결과 강조 출력 --
                st.markdown(f"<div style='font-size:18px; margin:12px 0;'><b>결절이 <span style='color:#222'>없을 확률</span>:</b> {negative_pct:.1f}%</div>", unsafe_allow_html=True)
                st.markdown(f"<div style='font-size:18px;'><b>결절이 <span style='color:#2dc9c8'>있을 확률</span>:</b> {positive_pct:.1f}%</div>", unsafe_allow_html=True)
                st.markdown(f"<div style='font-size:19px; margin:10px 0 0 0;'>→ <b>예측 결과: <span style='color:#ec4a7b'>{'결절 있음 (양성)' if pred_label==1 else '결절 없음 (음성)'}</span></b></div>", unsafe_allow_html=True)

                # 다운로드 버튼
                with open(mesh_path, "rb") as obj_file:
                    btn = st.download_button(
                        label="OBJ Download",
                        data=obj_file,
                        file_name=os.path.basename(mesh_path),
                        mime="application/octet-stream",
                        key="obj-download"
                    )
            except Exception as e:
                st.error(f"MeshCNN 입력 준비/예측 중 오류 발생: {e}")

        st.markdown("<br>", unsafe_allow_html=True)
st.empty()
with right:
    # 3D/2D 뷰어는 업로드 이후만 실제 결과 표시
    if uploaded_file is not None:
        st.subheader("🩻 CT 3D 뷰어")
        # 3D/2D용 verts/labels를 업로드 기반 데이터로 교체
        verts = np.load(TRAIN_VERT_PATH)   # (N, 3)
        labels = np.load(TRAIN_LABEL_PATH) # (N,) or (N, 1)
        features_7 = build_features_from_verts(verts)

        with torch.no_grad():
            features_tensor = torch.tensor(features_7, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            spiral_idx_batch = [idx.unsqueeze(0) for idx in spiral_idx]
            logits = model(features_tensor, spiral_idx_batch)
            logits = logits.detach().cpu().numpy()
            if logits.ndim == 2:
                logits = logits[0]
            preds = 1 / (1 + np.exp(-logits))
            pred_mask = preds > best_threshold

        #st.markdown("<div style='margin-bottom:10px; font-weight:700; font-size:22px;'>3D 뷰어</div>", unsafe_allow_html=True)
        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(verts[:, 0], verts[:, 1], verts[:, 2], c='gray', s=4, alpha=0.3, label='Lung All')
        if pred_mask.sum() > 0:
            ax.scatter(verts[pred_mask, 0], verts[pred_mask, 1], verts[pred_mask, 2], c='red', s=20, alpha=0.95, label='Predicted Nodule')
        #ax.set_title("DICOM zip 예측 결과", fontsize=15)
        ax.legend()
        plt.tight_layout()
        st.pyplot(fig, use_container_width=False)
        plt.close(fig)

        # 2D 뷰어 안내

        #st.markdown("<div style='margin-bottom:6px; font-weight:700; font-size:18px;'>2D 뷰어</div>", unsafe_allow_html=True)
        if "preprocessed" in st.session_state:
            st.divider()
            #st.header("🔎 결과 시각화")
            show_ct_viewer(st.session_state["preprocessed"]["image"])


    else:
        # 업로드 전에는 안내만 표시
        st.markdown("<div style='margin-bottom:10px; font-weight:700; font-size:22px;'>3D 뷰어</div>", unsafe_allow_html=True)
        st.markdown('<div class="viewer-box">여기에 3D 시각화가 표시됩니다.</div>', unsafe_allow_html=True)
        st.markdown("<div style='margin-bottom:10px; font-weight:700; font-size:22px;'>2D 뷰어</div>", unsafe_allow_html=True)
        st.markdown('<div class="viewer-box">여기에 2D 시각화가 표시됩니다.</div>', unsafe_allow_html=True)

# 안내문구는 기존대로 유지
st.markdown("""
<div class='center-announce'>
<br><br>
본 시스템은 연구용으로 제작된 시스템이며,<br>
정확한 진단은 반드시 전문 의료진의 판독을 참고하시기 바랍니다.
</div>
""", unsafe_allow_html=True)
