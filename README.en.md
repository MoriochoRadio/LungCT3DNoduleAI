# LungCT3DNoduleAI — AI-Based Lung CT 3D Visualization and Nodule Detection System

🇰🇷 [한국어](README.md) · 🇬🇧 English

> *AI-Based Lung Nodule Detection System Using Chest CT Images*
>
> A system where a patient uploads their own lung CT and it **reconstructs it in 3D and marks suspected nodule regions in color**. An undergraduate capstone design project that tried to tackle the problem — *"image reading takes days on average, and in the meantime the patient endures anxiety and an information gap"* — not by replacing diagnosis, but with *"a reference tool that fills the gap before and after diagnosis."* As **PM (Project Manager)** of the 6-person team **T.O.P**, I managed the schedule, progress, and task assignments, and handled data preprocessing. It is the deliverable of *Convergence Design and Project I (Capstone Design)* in the Department of Medical IT Engineering, 1st semester of 2025.

![Status](https://img.shields.io/badge/status-preserved-blue) ![Year](https://img.shields.io/badge/year-2025--1-blue) ![Role](https://img.shields.io/badge/role-PM%20%2B%20Data%20Preprocessing-orange) ![Stack](https://img.shields.io/badge/Python%203.9-PyTorch%20%2B%20Streamlit-success) ![Model](https://img.shields.io/badge/model-SpiralNet%20%2B%20PointNet%20%2B%20Transformer%20%2B%20MeshCNN-red) ![Dataset](https://img.shields.io/badge/dataset-LIDC--IDRI%20(TCIA)-yellow) ![Team](https://img.shields.io/badge/team-T.O.P%20(6%EC%9D%B8)-lightgrey)

![Title](assets/screenshots/title.png)

> Final presentation cover — **T.O.P** (Technology Of Prognosis, *"prediction technology"*) · 2025.06.16 · Ver.1.0.0

---

## ⚠ About Build/Run Status

This repo **preserves the code exactly as it was at the June 2025 presentation**. Unlike its sister project [`MedQueue`](https://github.com/MoriochoRadio/MedQueue) (unbuildable due to Xamarin's end of support), this project runs on Python 3.9.21 + PyTorch + Streamlit and was runnable at the time of writing.

To run it yourself, however, note the following:
- **Training data is not included.** You must download LIDC-IDRI (public lung CT-DICOM from TCIA) yourself ([download guide](#-dataset--training--performance)).
- **Model weights** (`models/dhkstjd.pth`, 52 MB) are pushed via Git LFS. Run `git lfs pull` after cloning.
- **Paths inside the code** are the absolute paths of the shared school PC used for training (`C:/Users/<PC_A>/Desktop/spiral_torch/...`), masked as `<PC_A>`; adjust them to your own environment.
- Run: `streamlit run src/test2.py`

---

## 📚 Project Context

| Item | Details |
|---|---|
| **When** | 1st semester, 2025 academic year (presented 2025.06.16) |
| **Affiliation** | Dept. of Medical IT Engineering, Konyang University |
| **Course** | Convergence Design and Project I (Capstone Design), 3rd year, section 1 |
| **Advisor** | Prof. Song ◯◯ |
| **Team** | **T.O.P** (Technology Of Prognosis, *"prediction technology"*) — 6 members |
| **Members** | **PM Kim TaeKyoung (me)** / CM ◯◯◯ / QA1 ◯◯◯ / QA2 ◯◯◯ / ENG1 ◯◯◯ / ENG2 ◯◯◯ |
| **My (KimTaeKyoung) role** | ★ **PM (overall project lead · schedule management · progress management · task assignment) + data preprocessing (DICOM → mesh → vertex)** |
| **My first-author deliverables** | 5 — team formation document · project proposal · initial development plan · development completion report · MS Project |
| **AI model / Streamlit UI** | mainly ENG1 / ENG2 (not my area — see [*My Contributions*](#-my-contributions-kimtaekyoung--pm--data-preprocessing) below) |

> 📌 This is the **first project in my portfolio where the "AI" part was completed**. Where the earlier [`MedQueue`](https://github.com/MoriochoRadio/MedQueue) (patient waiting), [`SchoolbusRFID`](https://github.com/MoriochoRadio/SchoolbusRFID) (child safety), and [`ElderCaringApp`](https://github.com/MoriochoRadio/ElderCaringApp) (elder care) were *apps and IoT*, this project is my *first full-scale deep learning medical imaging* project and my *first PM experience*.

---

## ❓ Problem Definition

![Problem](assets/screenshots/problem_2_waiting.png)

This project started from three problems surrounding lung CT diagnosis:

1. **Lung cancer: #1 in cancer mortality** — lung cancer is the leading cause of cancer death in Korea. Early detection strongly determines survival.
2. **Patients cut off from information** — *"Patients cannot understand CT results without a clinician's explanation, and image reading takes several days or more on average. The helplessness and information gap experienced while waiting for results worsen the patient's care experience."* (citing O'Sullivan et al. 2017, Munn et al. 2014, Lee & Kim 2023)
3. **No patient-facing systems** — existing image reading systems are all for clinicians. There is hardly any tool a patient can use to see for themselves *"what state my CT is in."*

---

## 💡 Solution

![Solution](assets/screenshots/solution.png)

> *"Provide an aid that helps patients decide what to do before and after a lung CT diagnosis."*

**Not replacing diagnosis, but filling the gap before and after it as a reference tool** — that is the core positioning of this project. User scenario:

1. Compress the CT-DICOM files into a zip and upload it to the website
2. The system unzips it and processes the CT data
3. The screen shows the lung CT images + analysis results + a 3D image
4. **Suspected nodule regions are marked in color on the 3D image**
5. Save the results to a file for later use

![User Scenario](assets/screenshots/user_scenario.png)

### 3D Visualization — Reading Nodules by Color

| Color | Meaning |
|---|---|
| Gray points (Lung BG) | lung structure (not a prediction target) |
| Yellow points (GT Nodule) | actual nodule locations (ground truth) |
| Red points (Predicted) | nodule locations predicted by the model |
| Orange points (Correct) | locations the model predicted that are actually nodules |

---

## 🛠️ Tech Stack

![Tech Stack](assets/screenshots/tech_stack.png)

| Area | Technology |
|---|---|
| **Language / environment** | Python 3.9.21 · Jupyter Lab |
| **Deep learning** | PyTorch (SpiralNet + PointNet + Transformer + MeshCNN 4-branch hybrid) |
| **Medical imaging** | pydicom (DICOM loading) · trimesh + scikit-image marching cubes (mesh generation) · scipy (3D resampling) |
| **Web UI** | Streamlit |
| **Visualization** | matplotlib · 3D point cloud rendering |

See [`requirements.txt`](requirements.txt) for full dependencies. The file was auto-generated by analyzing the `import` statements in the code.

---

## 🏗️ System Design

![System Architecture](assets/architecture/system_architecture.png)

The overall pipeline is *"DICOM upload → preprocessing → 3D mesh conversion → 4-branch model inference → 3D visualization."*

```
┌─────────────────────────────────────────────────────────────────┐
│  User: uploads CT-DICOM zip (Streamlit web)                     │
└────────────────────────────┬────────────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  ★ Data preprocessing (my area: src/test2.py, 12 fns, 247 ln)   │
│  find_dicom_files → load_scan → get_pixels_hu (HU conversion)   │
│  → resample (spacing correction) → segment_lung_mask            │
│    (marching cubes)                                             │
│  → mesh generation → vertex 7-feature → .npy                    │
└────────────────────────────┬────────────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  AI model (designed/trained by ENG): 4-branch hybrid            │
│  ┌──────────────┬──────────────┬──────────────┬───────────────┐ │
│  │ ImprovedSpiral│   PointNet   │  Transformer │   MeshCNN     │ │
│  │ Net (mesh conv│  (point      │ (self-       │  (edge        │ │
│  │ + SE + skip)  │   feature)   │  attention)  │   feature)    │ │
│  └──────┬───────┴──────┬───────┴──────┬───────┴───────┬───────┘ │
│         └──── ensemble (spiral+pointnet)/2 ────┘ + classification│
└────────────────────────────┬────────────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  3D visualization (suspected nodules in color) + result saving  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🤖 Model Architecture

![Performance](assets/screenshots/performance_metrics.png)

The 4-branch hybrid model (`HybridSpiralNetPointNetTransformer`) detects lung nodules from four perspectives — mesh / point cloud / attention / edge:

| Branch | Class | Role |
|---|---|---|
| 1 | `ImprovedSpiralNet` (68 lines) | multi-scale SpiralConv encoder-decoder + Squeeze-Excitation + skip connections |
| 2 | PointNet MLP | point features → global pooling |
| 3 | `SimplePointTransformerBlock` | multi-head self-attention (heads=4) |
| 4 | `MeshCNN` | mesh edge features + nodule presence classification |

Training used `HybridLoss` (focal 0.7 / dice 0.3), which combines `AdaptiveFocalLoss`, Dice Loss, and Hard Negative Mining to cope with class imbalance.

> ⚠ **Honest disclosure**: The design and training of the model architecture above were **mainly handled by ENG1 / ENG2**. My (PM) code area was data preprocessing — see [*My Contributions*](#-my-contributions-kimtaekyoung--pm--data-preprocessing) below for the detailed division of work.

---

## ✨ Key Features / UI

| Streamlit main screen | Result screen |
|---|---|
| ![UI Main](assets/screenshots/ui_streamlit_main.png) | ![UI Result](assets/screenshots/ui_streamlit_result.png) |

### Demo Videos

<video src="assets/videos/demo_main.mp4" controls width="800"></video>

> If the video above does not display, play [`assets/videos/demo_main.mp4`](assets/videos/demo_main.mp4) directly (1920×1080, 0:33 — the video embedded in presentation slide 17).
> An actual recording of Streamlit in use is at [`assets/videos/demo_streamlit_recording.mp4`](assets/videos/demo_streamlit_recording.mp4) (1:26).

---

## 📊 Dataset / Training / Performance

![Dataset](assets/screenshots/dataset_lidc_tcia.png)

### Dataset — LIDC-IDRI (TCIA)

> *"Used the lung CT-DICOM image dataset released by the U.S. National Cancer Institute (TCIA). Train : validation : test ratio set to 8:1:1."*

- **Dataset**: [LIDC-IDRI](https://www.cancerimagingarchive.net/collection/lidc-idri/) (Lung Image Database Consortium) — hosted by TCIA (The Cancer Imaging Archive), released by the U.S. National Cancer Institute (NCI)
- **Split**: train : val : test = 8 : 1 : 1
- **Preprocessing**: DICOM → HU conversion → lung segmentation (marching cubes) → 3D mesh → vertex feature `.npy` (★ my area)
- **Training**: Google Colab, 46+ epochs (loss/F1 evolution preserved in [`notebooks/training_log.txt`](notebooks/training_log.txt))

> 📌 The dataset is not included in the repo due to size and licensing. Download it directly from the TCIA link above and run the preprocessing code. Only a minimal sample pair for verifying that the code works is included in [`data_sample/`](data_sample/) (35 KB).

### Performance — An Honest Self-Assessment

> *"Got 171 of 234 test cases right, for 73.1% accuracy. Predicted well on data with actual nodules — 160 out of 176. On data without nodules, performance was low at 11/58. **The data imbalance needs improvement.**"* (presentation slide 15, delivered by me)

| Metric | Value |
|---|---|
| Accuracy | 73.1% (171/234) |
| Positive recall | 90.9% (160/176) |
| **Negative recall** | **19.0% (11/58)** ← ★ data imbalance |
| Best F1 | 0.78 |

A negative recall of 19% means the model often *"predicted a nodule where there was none."* It is a direct reflection of the class imbalance — positive cases (176) outnumbered negatives (58) three to one — and in the presentation I acknowledged the limitation myself: *"the data imbalance needs improvement."* An honest landing point for a first undergraduate AI project.

---

## 👤 My Contributions (KimTaeKyoung · PM + Data Preprocessing)

Since this was a team project, I clearly separate the parts I handled from the parts my teammates handled. Where the earlier projects (MedQueue/SchoolbusRFID/ElderCaringApp) were *"an engineer owning a specific module,"* this project was my first experience coordinating an entire team as **PM of a 6-person team**.

### Track 1 — The Scale of PM Responsibility (Proven by Data)

![Team Org](assets/architecture/team_org.png)

On the org chart I was in charge of **overall project lead · task assignment and schedule management · progress management**. That responsibility is stamped into the deliverables as data:

| Metric | Value |
|---|---|
| Pages of the 166-page integrated deliverables naming *"Kim TaeKyoung"* | **143 pages (86%)** |
| Of those, pages marked *"PM Kim TaeKyoung"* (explicitly responsible) | **129 pages (78%)** |

My name appears on 86% of the 166 pages of deliverables, and the responsible-person notation *"PM Kim TaeKyoung"* on 78%. It shows the PM role was not a mere title but was consistently recorded as the responsible party across all deliverables.

### Track 2 — Five First-Author Deliverables

![Deliverables](assets/screenshots/deliverables_table.png)

Of the 21 deliverables, **I wrote 5 as first author**:

| Deliverable | Pages |
|---|---|
| Team formation document (v2.0.0) | 4 |
| Project proposal (v2.0.0) | 5 |
| Initial development plan (v1.0.0) | 9 |
| Development completion report (v2.0.0) | 9 |
| MS Project (schedule management) | (separate) |

These 4 written deliverables (27 pages) are extracted into one file, [`docs/deliverables_authored_by_kim_taekyoung.pdf`](docs/deliverables_authored_by_kim_taekyoung.pdf) (teammate PII masked). The MS Project schedule file is in .mpp format and archived separately; the Gantt chart can be seen in [`assets/screenshots/gantt_slide19.png`](assets/screenshots/gantt_slide19.png).

### Track 3 — Data Preprocessing (Honest Ambiguity)

My code area was **data preprocessing (the DICOM → mesh → vertex `.npy` conversion pipeline)**. [`src/test2.py`](src/test2.py) contains the 12 related functions (247 lines total):

```
find_dicom_files       (recursive DICOM file search)
load_scan              (load DICOM slices + sort by ImagePositionPatient)
get_pixels_hu          (HU, Hounsfield Unit conversion)
resample               (3D zoom, spacing correction)
segment_lung_mask      (marching cubes lung segmentation)
build_features_from_verts  (build vertex 7-features)
mesh_load_scan / mesh_get_pixels_hu / mesh_resample / mesh_segment_lung_mask
mesh_lung_mask_to_mesh (marching cubes → trimesh)
preprocess_mesh_single (136 lines — MeshCNN input preprocessing, the largest function)
```

### The Areas I Did Not Handle (Honest Attribution, Confirmed by Data)

**The design and training of the AI model (4-branch hybrid) were mainly handled by ENG1 / ENG2.** This is not a guess but confirmed by data — in all 5 cells of the training notebook [`notebooks/Untitled0.ipynb`](notebooks/Untitled0.ipynb), Google Colab's `executionInfo.user` metadata **automatically recorded the author/executor as ENG1 (◯◯◯)** (that metadata has been masked). The main Streamlit UI (the web page portion of `test2.py`) was also an ENG responsibility per the org chart.

As PM I was responsible for the whole team's schedule and deliverables, while entrusting the core technical area of AI model training to the engineer teammates and supporting them with data preprocessing — that is my honest position in this project.

---

## 📅 Project Schedule

![Gantt](assets/screenshots/gantt_slide19.png)

The schedule I created and managed in MS Project (presentation slide 19). From the March 2025 kickoff to the June presentation, I managed data collection/preprocessing, model design/training, Streamlit integration, and presentation preparation stage by stage over one semester.

---

## 🧪 Expected Impact + Limitations

![Expected Effects](assets/screenshots/expected_effects.png)

### Expected Impact (presentation slide 18)
1. Patients can check for themselves, easing anxiety while waiting for results
2. Usable as reference information before and after hospital diagnosis
3. A practical reference tool that fills the gap, rather than replacing diagnosis

### Limitations (honestly)
- **Negative recall 19.0%** — data imbalance problem (see [Performance](#-dataset--training--performance) above)
- **Training centered on a single case type** — a wider variety of cases is needed
- Future development plans from presentation slide 21: *"Improve the low 'no nodule' prediction accuracy / secure more lung CT data / expand beyond the lungs to other organs such as the liver and brain."*

---

## 📂 Archive

| Material | Path | Notes |
|---|---|---|
| 🎤 Final presentation (44 slides) | [`docs/presentation_final_redacted.pptx`](docs/presentation_final_redacted.pptx) | Git LFS |
| 🪧 Panels (achievement exhibition) | [`docs/panel_v1_redacted.pptx`](docs/) · panel_v2 | Git LFS |
| 📑 Development completion report | [`docs/final_report_redacted.pdf`](docs/final_report_redacted.pdf) | Hancom HWP→PDF conversion |
| 📑 Integrated deliverables volume (166p) | [`docs/deliverables_full_redacted.pdf`](docs/deliverables_full_redacted.pdf) | teammate PII masked |
| ★ 📖 My first-author deliverables (27p) | [`docs/deliverables_authored_by_kim_taekyoung.pdf`](docs/deliverables_authored_by_kim_taekyoung.pdf) | PM formation doc + proposal + plan + completion report |
| 🤖 Main model weights | [`models/dhkstjd.pth`](models/) | Git LFS (52 MB) |
| 📓 Training notebook (Colab) | [`notebooks/Untitled0.ipynb`](notebooks/Untitled0.ipynb) | written by ENG1 (confirmed via metadata) |
| 📓 Inference/visualization notebook | [`notebooks/완성이다.ipynb`](notebooks/) | original Korean filename preserved |
| 📈 Training log (46 epochs) | [`notebooks/training_log.txt`](notebooks/training_log.txt) | loss/F1 evolution |
| 🎬 Demo videos ×2 | [`assets/videos/`](assets/videos/) | demo_main (0:33) + streamlit_recording (1:26) |
| 🧩 Sample for running the code | [`data_sample/`](data_sample/) | one A0002 pair (35 KB) |

> 📌 The names of the 5 teammates and 1 professor other than myself are masked as `◯◯◯` in the presentation materials, deliverables, and metadata. The training data (645 MB) and the original MS Project file (.mpp) are kept in `docs/_archive/local-only/` and are not pushed to GitHub. Teammates' faces in the meeting photos inside the deliverable PDFs were also removed.

---

## 📚 References

Key sources cited by this project (presentation slides 7, 22-23):
- On patient information gaps and waiting anxiety: O'Sullivan et al. (2017), Munn et al. (2014), Lee & Kim (2023)
- Lung cancer mortality statistics: National Cancer Information Center · Statistics Korea
- Dataset: Armato et al., *The Lung Image Database Consortium (LIDC-IDRI)*, Medical Physics (2011)

---

## ✍️ Retrospective

This project is the point where the *"medical AI"* line was first completed. Where [MedQueue](https://github.com/MoriochoRadio/MedQueue), [SchoolbusRFID](https://github.com/MoriochoRadio/SchoolbusRFID), and [ElderCaringApp](https://github.com/MoriochoRadio/ElderCaringApp) were the *"app"* part of the medical/care domain, T.O.P is the first project where I coordinated a 6-person team as PM and brought the *"AI"* part in, in earnest.

My areas were the PM work — schedule, progress, and task assignment management — and data preprocessing (the DICOM → mesh → vertex `.npy` conversion). The 4-branch hybrid model itself (SpiralNet + PointNet + Transformer + MeshCNN) was designed and trained by ENG1/ENG2, and that fact is automatically fossilized as the author in the training notebook's Colab metadata. I was first author of 5 of the 21 deliverables, and I am recorded as the responsible PM on 143 of the 166 pages (86%).

The gap between 73.1% accuracy and 19.0% negative recall on the 234-case test set lays bare the data imbalance problem of a first undergraduate AI project. In the presentation I acknowledged it myself — *"the data imbalance needs improvement"* — and that awareness of limitations carried over into the next project.

What is interesting is that both the *previous project* and the *next project* are already fossilized inside this presentation deck. Slide 40 names *"2024 Convergence Design II — MAC medical twin pulmonary vessel model"* as the 3D modeling technology base of this project, and slide 44 (system module structure) already includes a *"liver nodule detection model module"* — the starting point of the next project, [AILungandLiver](https://github.com/MoriochoRadio/AILungandLiver) (the lung/liver extension). The **3-step evolution of medical imaging AI (pulmonary vessels MAC → lung nodules T.O.P → lung/liver nodules AILungandLiver)** is thus fossilized within a single deck. The thread later continues on to the [seed-project](https://github.com/MoriochoRadio/seed-project) capstone and [sperm-ai](https://github.com/MoriochoRadio/sperm-ai).

The experience of taking on PM for the first time as an undergraduate — coordinating six people's schedules and deliverables — and learning *"how to honestly distinguish what I did from what I did not do"* will, I think, stay with me longer than the code from this project.

---

## 🗂️ Related Repositories

### The Foundations That Made This Project Possible
- [`study-java`](https://github.com/MoriochoRadio/study-java) · [`study-windows-programming`](https://github.com/MoriochoRadio/study-windows-programming) — basic programming study

### Same Healthcare/Care Domain (Apps · IoT)
- [`MedQueue`](https://github.com/MoriochoRadio/MedQueue) — real-time hospital waiting info app (2024-1)
- [`SchoolbusRFID`](https://github.com/MoriochoRadio/SchoolbusRFID) — location-based child drop-off tagging (2024-2)
- [`ElderCaringApp`](https://github.com/MoriochoRadio/ElderCaringApp) — health monitoring for seniors living alone (2024-2)

### ★ Medical Imaging AI Line (3-Step Evolution)
- **MAC** (2024-II) — medical twin pulmonary vessel model *(this project's 3D technology base, fossilized on PPT slide 40)*
- **LungCT3DNoduleAI** (2025-1, this project) — lung CT nodule detection *(me: PM)*
- [`AILungandLiver`](https://github.com/MoriochoRadio/AILungandLiver) — lung/liver nodule extension *(the next project, fossilized on PPT slide 44)*
- [`sperm-ai`](https://github.com/MoriochoRadio/sperm-ai) — AI sperm motility analysis
- [`seed-project`](https://github.com/MoriochoRadio/seed-project) — Team SEED capstone (in progress)

---

## License

See [`LICENSE.md`](LICENSE.md). This repo publishes the learning outcome of an undergraduate team project for portfolio purposes. The LIDC-IDRI dataset follows TCIA's [Creative Commons Attribution 3.0](https://creativecommons.org/licenses/by/3.0/) license. Commercial use requires team agreement.

---

*Author: [MoriochoRadio](https://github.com/MoriochoRadio) (KimTaeKyoung) · Dept. of Medical IT Engineering, Konyang University · Team T.O.P PM*
