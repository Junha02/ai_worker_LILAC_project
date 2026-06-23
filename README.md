# Adaptive Assistive Robot Control with LILAC

Language-Informed Latent Actions with Corrections를 SH5 right arm teleoperation에 적용한 Human-Robot Interaction(HRI) 프로젝트입니다. 사용자는 **2-DoF Vader5 joystick**으로 latent action `z`만 입력하고, LILAC model은 현재 robot/object state와 active language command를 함께 해석해 **6-DoF end-effector action**을 생성합니다.

일반적인 joystick teleoperation은 제한된 input DoF 때문에 position과 orientation을 번갈아 조작해야 합니다. 이 프로젝트는 instruction/correction language를 이용해 같은 joystick input의 의미를 상황에 맞게 바꾸고, 사용자가 mode switching 없이 cup-to-bowl, remote-controller-to-box 같은 task를 수행하도록 돕는 shared-autonomy controller를 구현합니다.

## Table of Contents

- [1. 핵심 아이디어](#core-idea)
- [2. 현재 프로젝트 구성](#project-pipeline)
- [3. Language 처리](#language-processing)
- [4. System Architecture](#system-architecture)
- [5. Data Collection and Training](#data-training)
- [6. Safety and HRI Details](#safety-hri)
- [7. Dataset](#dataset)
- [8. Troubleshooting](#troubleshooting)
- [9. Repository Structure](#repository-structure)

<a id="core-idea"></a>

## 1. 핵심 아이디어

![기존 assistive robot control의 한계](docs/images/motivation-control-limitations.png)

![MuJoCo simulation과 joystick 조작](docs/images/simulation-demo.png)

사용자는 joystick의 2D 입력을 직접 6D Cartesian command로 매핑하지 않습니다. 대신 joystick 입력은 latent action `z`가 되고, LILAC decoder가 active language와 state를 조건으로 `z`를 6-DoF action으로 해석합니다.

```text
(state s, utterance u, alpha, latent joystick input z) -> predicted 6-DoF action a'
```

예를 들어 같은 joystick 방향이라도 active utterance가 `right`이면 오른쪽 이동을, `pour water`이면 컵을 기울이는 회전 동작을 더 강하게 생성할 수 있습니다. 사용자는 2-DoF 입력만 유지하면서 task instruction과 correction stack을 통해 더 풍부한 robot motion을 얻습니다.

![LILAC model 개념도](docs/images/lilac-model-overview.png)

### Model input과 output

| Symbol | 의미 | 현재 구현 |
| --- | --- | --- |
| `s` | robot/task current state | right arm joint 7D + end-effector pose 6D + object position 6D = 19D |
| `u` | instruction 또는 correction utterance | canonical utterance의 768D SBERT embedding |
| `alpha` | instruction/correction gating label | instruction `1.0`, correction `0.0` |
| `z` | joystick이 제어하는 latent action | 2D |
| `a`, `a'` | target / predicted robot action | `[dx, dy, dz, droll, dpitch, dyaw]` 6D |

### Conditional Autoencoder 구조

Training 단계에서는 demonstration action `a`를 compressor가 2D latent action `z`로 압축하고, decoder가 `(s, u, alpha, z)`로부터 `a'`를 복원합니다. Inference 단계에서는 compressor를 쓰지 않고, 사람이 joystick으로 넣은 `z`를 decoder에 바로 전달합니다.

```text
Training
  (s, u, alpha, a) -> Action Encoder -> z
  (s, u, alpha, z) -> Decoder        -> a'
  loss = MSE(a', a)

Inference
  human joystick -> z
  (s, u, alpha, z) -> Decoder -> a'
```

Decoder는 state와 language를 FiLM으로 결합한 뒤, Gram-Schmidt로 orthonormalized된 두 개의 6-DoF basis를 생성합니다.

```text
B(s, u, alpha) in R^(6x2)
a' = B(s, u, alpha) @ z
```

`alpha=1`인 instruction은 state context를 적극적으로 사용하고, `alpha=0`인 correction은 state 의존성을 줄여 `right`, `up`, `pour water` 같은 즉각적인 수정 의도를 우선합니다.

<a id="project-pipeline"></a>

## 2. 현재 프로젝트 구성

현재 repo는 **MuJoCo simulation notebook + Python package + training/runtime scripts** 중심으로 구성되어 있습니다. 주요 entrypoint는 `sim_notebook/`의 5개 notebook입니다.

| Notebook | 역할 |
| --- | --- |
| `00_lilac_collect_ik_sh5.ipynb` | Direct IK 기반 SH5 right-arm data collection |
| `01_lilac_collect_fd_sh5.ipynb` | Forward dynamics 기반 data collection |
| `02_lilac_train_sh5.ipynb` | recorded trajectory를 training array로 변환하고 LILAC model 학습 |
| `03_lilac_inference_2dof_sh5.ipynb` | Vader5 2-DoF joystick + manual hand control inference |
| `04_lilac_inference_fd_sh5.ipynb` | forward-dynamics inference, runtime language update, haptic/contact 실험 |

CLI script는 notebook에서 반복되는 전처리와 학습 작업을 분리하기 위해 제공합니다.

```text
scripts/precompute_language_index.py      # canonical utterance SBERT index 생성
scripts/prepare_training_arrays.py        # recorded episodes -> training arrays
scripts/train_lilac.py                    # arrays -> runs/lilac_sh5_right model bundle
scripts/runtime_language_cli.py           # inference 중 language command를 file queue에 append
scripts/label_alphas.py                   # canonical alpha label 확인/보조
```

<a id="language-processing"></a>

## 3. Language 처리

사람은 같은 의도를 여러 방식으로 말합니다. 본 프로젝트는 [`data/language/lilac_canonical_utterances.json`](data/language/lilac_canonical_utterances.json)에 robot이 실행할 수 있는 canonical command를 정의하고, runtime utterance를 이 command 중 하나로 매핑합니다.

1. 입력 문장을 normalize한 뒤 canonical text, id, alias와 exact match를 먼저 수행합니다.
2. Exact match가 성공하면 Gemini 호출 없이 local result로 처리합니다.
3. 새로운 표현은 Gemini selector가 canonical id 중 하나를 선택합니다.
4. 선택된 id는 parser가 다시 검증하고, kind에 따라 instruction 또는 correction으로 stack에 반영됩니다.

![Exact Match와 Gemini 분기 구조](docs/images/language-exception-handling.png)

### Instruction과 Correction stack

- `instruction`: 전체 task를 설명합니다. 새 instruction이 들어오면 correction stack은 초기화됩니다.
- `correction`: 현재 동작을 즉시 수정합니다. LIFO stack에 push됩니다.
- `pop`: 가장 최근 correction을 제거하고 이전 correction 또는 instruction으로 돌아갑니다.
- `clear`: correction stack만 비웁니다.

현재 canonical command는 다음과 같습니다.

| Type | Canonical utterance |
| --- | --- |
| instruction | `Pick up the cup and pour water into the bowl.` |
| instruction | `pick up remote controller put in box` |
| correction | `right`, `left`, `up`, `down`, `front`, `back`, `pour water` |

Runtime inference에서는 `data/runtime_language_command.txt`를 file-backed command queue로 사용합니다. 별도 terminal에서 다음 CLI를 실행하면 inference notebook이 비동기적으로 command를 읽습니다.

```bash
python3 scripts/runtime_language_cli.py
```

`.env`에 `GEMINI_API_KEY` 또는 `GOOGLE_API_KEY`를 넣으면 free-form utterance를 Gemini로 canonicalize할 수 있습니다. `.env`는 repository에 올리지 않도록 `.gitignore`에 포함되어 있습니다.

<a id="system-architecture"></a>

## 4. System Architecture

![전체 System Architecture](docs/images/system-architecture.png)

현재 구현의 runtime flow는 다음과 같습니다.

```text
Vader5 joystick z
        +
active language stack
        +
SH5/object state
        |
        v
LILACSharedAutonomyController
        |
        v
6-DoF end-effector delta
        |
        v
SH5 right-arm IK / MuJoCo simulation target
```

핵심 모듈은 `package/`에 있습니다.

| Module | 역할 |
| --- | --- |
| `lilac_model.py` | Conditional autoencoder, state/language encoder, basis projection, compressor |
| `controller.py` | language stack + latent `z` -> 6-DoF EE delta runtime controller |
| `language.py` | canonical utterance dataset, Gemini selector, SBERT language index |
| `data.py` | trajectory recorder, episode metadata, training array construction |
| `training.py` | PyTorch training loop, language index build, latent alignment 저장 |
| `sh5_right_arm.py` | SH5 right-arm IK, MuJoCo scene/object setup, Vader5 mapping |
| `notebook_units.py` | notebooks에서 공유하는 setup/runtime helper |
| `real_hri.py`, `real_zed.py`, `real_stt.py`, `real_sh5_zmq.py` | real-robot HRI, ZED RGB, Whisper STT, ZMQ/SH5 helper |

<a id="data-training"></a>

## 5. Data Collection and Training

Data collection notebook은 task별로 demonstration episode를 저장합니다. 각 episode는 metadata `.json`과 frame data `.npz` 한 쌍으로 구성됩니다.

```text
data/<task>/<instruction or correction>/<episode_id>.json
data/<task>/<instruction or correction>/<episode_id>.npz
```

Training workflow는 다음 순서입니다.

```bash
python3 scripts/precompute_language_index.py
python3 scripts/prepare_training_arrays.py
python3 scripts/train_lilac.py
```

또는 `sim_notebook/02_lilac_train_sh5.ipynb`에서 같은 흐름을 notebook으로 실행할 수 있습니다. 학습 결과는 기본적으로 `runs/lilac_sh5_right/`에 저장되며, repository에는 생성 산출물이 올라가지 않도록 `runs/`가 ignore되어 있습니다.

Model bundle은 다음 파일을 포함합니다.

```text
runs/lilac_sh5_right/
├── model.pt
├── config.json
├── history.json
├── language_index.npz
└── latent_alignment.npz
```

`latent_alignment.npz`는 demonstration에서 학습된 latent space와 실제 joystick `z`의 sign/swap/rotation ambiguity를 줄이기 위한 linear calibration입니다.

<a id="safety-hri"></a>

## 6. Safety and HRI Details

### Safe demonstration만 저장

Data collection 중 잘못된 방향으로 움직이거나 collision 위험이 발생하면 joystick의 A button으로 recording을 cancel하고 simulation target을 reset합니다. 실패 trajectory가 dataset에 섞이지 않도록 하고, 안전하게 수행된 demonstration만 학습에 사용합니다.

Collection 단계에서 B button은 recording start/save toggle로 사용됩니다. Inference 단계에서는 A button이 simulation reset, B button이 가장 최근 correction을 `pop`하는 역할을 합니다.

![Joystick 기반 Data Collection](docs/images/data-collection.png)

![A button Simulation Reset](docs/images/simulation-reset.png)

### Robot contact만 Haptic feedback으로 전달

MuJoCo contact 전체를 haptic feedback에 바로 연결하면 cup-floor/table contact처럼 정상적인 object contact에서도 joystick vibration이 발생합니다. 이를 피하기 위해 robot body prefix가 포함된 contact만 filtering하고, robot arm/hand가 주변 물체와 충돌할 때만 feedback을 전달합니다.

![Robot contact 기반 Haptic Feedback](docs/images/haptic-feedback.png)

### Real-robot HRI helper

`real_hri.py`, `real_zed.py`, `real_stt.py`, `real_sh5_zmq.py`는 notebook에서 실제 SH5 환경과 연결할 때 사용하는 helper입니다. Head target message, ZED RGB subscriber, Whisper-small STT, SH5 joint message layout, body skeleton/debug drawing 등을 제공합니다.

<a id="dataset"></a>

## 7. Dataset

현재 repository에는 수집된 trajectory와 학습용 array가 포함되어 있습니다.

```text
data/
├── cup_to_bowl/
│   ├── instruction/
│   └── correction/
├── remote_controller_to_box/
│   └── instruction/
├── language/
│   ├── lilac_canonical_utterances.json
│   └── language_index.npz
└── training/
    └── lilac_sh5_right_arrays.npz
```

각 `.npz` episode에는 joint state, end-effector pose, 6-DoF action, latent `z`, active utterance, correction stack, contact flag 등이 저장됩니다.

| Task / Type | Utterance | Episodes | Frames |
| --- | --- | ---: | ---: |
| `cup_to_bowl/instruction` | `Pick up the cup and pour water into the bowl.` | 15 | 6,703 |
| `cup_to_bowl/correction` | `right`, `left`, `up`, `down`, `front`, `back`, `pour water` | 70 | 4,895 |
| `remote_controller_to_box/instruction` | `pick up remote controller put in box` | 12 | 6,429 |
| **Total** | 9 canonical utterances | **97** | **18,027** |

Consecutive frame pair로 action을 구성한 최종 training array에는 **17,930 samples**가 포함되어 있습니다.

<a id="troubleshooting"></a>

## 8. Troubleshooting

### 1. Translation은 충분하지만 pouring rotation이 약한 문제

수집 trajectory에는 `x, y, z` translation movement가 많고, orientation 중에서는 물을 붓기 위한 `roll`이 핵심입니다. 초기 inference에서는 limited 2D latent capacity와 reconstruction objective가 빈도가 높고 변화량이 큰 translation pattern을 우선적으로 표현하면서, task-critical roll output의 effective magnitude가 부족했습니다.

이를 보정하기 위해 inference notebook에서 position과 rotation scale을 분리했습니다. 현재 inference runtime은 `action_pos_scale`, `action_rot_scale`을 따로 지정해 model이 학습한 action direction은 유지하면서 rotation channel의 contribution을 조절합니다.

### 2. 모든 MuJoCo contact에서 joystick이 진동하는 문제

전체 contact를 haptic signal로 사용하지 않고, robot body prefix가 포함된 contact만 남기는 filtering logic을 사용합니다. Object-floor contact는 무시하고 robot collision만 사용자에게 전달합니다.

### 3. Free-form language가 정의되지 않은 command를 만드는 문제

Exact match를 우선하고, novel utterance만 Gemini로 전달합니다. Gemini output도 canonical JSON id 중 하나로 제한하고 parser에서 다시 검증해 hallucinated command 실행을 차단합니다.

### 4. Runtime command가 inference에 반영되지 않는 문제

Inference notebook과 CLI가 같은 `data/runtime_language_command.txt`를 보고 있는지 확인해야 합니다. Notebook이 runtime을 새로 load하면 command file header가 다시 작성되므로, inference cell 실행 뒤 CLI에서 command를 입력하는 순서가 안전합니다.

<a id="repository-structure"></a>

## 9. Repository Structure

```text
.
├── asset/                  # MuJoCo task object mesh와 scene asset
├── data/                   # trajectory, canonical language, training arrays
├── docs/images/            # README 이미지
├── package/
│   ├── lilac_model.py      # Conditional Autoencoder 기반 LILAC model
│   ├── language.py         # Exact match, Gemini selector, language stack, SBERT index
│   ├── controller.py       # latent z -> 6-DoF action runtime controller
│   ├── data.py             # episode recorder와 training array builder
│   ├── training.py         # model training과 latent alignment
│   ├── sh5_right_arm.py    # SH5 right arm IK와 Vader5 control utility
│   ├── notebook_units.py   # notebook-facing runtime/setup helper
│   ├── real_hri.py         # HRI trigger/head target/face helper
│   ├── real_zed.py         # ZED RGB subscriber
│   ├── real_stt.py         # Whisper-small STT helper
│   └── real_sh5_zmq.py     # SH5 ZMQ and skeleton utilities
├── scripts/                # language index, preprocessing, training, runtime CLI
└── sim_notebook/           # collection, training, inference workflow
```
