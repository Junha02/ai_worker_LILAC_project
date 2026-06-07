# LILAC ROS 2 Workspace

기존 `package/`의 language dataset과 LILAC model loading path를 ROS 2 node로 연결한 workspace입니다.

## Packages

- `lilac_interfaces`: active language, latent input, state, action, q_pos, contact, rumble message와 utterance service
- `lilac_ros`: language manager, latent input, policy, IK bridge, mock simulation, haptic node

## Topic / Service Flow

```text
/lilac/apply_utterance (Service)
    -> /lilac/active_language

/joy -> /vader5/latent_z
/lilac/state + /lilac/active_language + /vader5/latent_z
    -> /lilac/ee_delta_6d
    -> /sh5/qpos_cmd
    -> /lilac/state

/sim/contact_info -> /vader5/rumble
```

`policy` node는 `runs/lilac_sh5_right/`의 trained model을 우선 로드하고, artifact가 준비되지 않은 환경에서는 ROS message flow를 확인할 수 있는 demo basis policy를 사용합니다. `mock_sim`과 `ik_bridge`는 실제 MuJoCo/SH5 process로 교체할 수 있는 adapter입니다.

## Local ROS 2 Build

```bash
cd ros_ws
source /opt/ros/humble/setup.bash
colcon build --symlink-install
source install/setup.bash
ROS_LOCALHOST_ONLY=1 ros2 launch lilac_ros lilac_demo.launch.py
```

Utterance service 예시:

```bash
ros2 run lilac_ros utterance_client "pour water"
```

## Docker

전체 node를 하나의 container에서 실행:

```bash
docker build -t lilac-ros2:humble -f docker/Dockerfile .
docker run --rm --network host --ipc host lilac-ros2:humble
```

PDF architecture처럼 language, controller, simulation container를 분리:

```bash
docker compose up --build
```

Compose 구성은 동일 host network와 `ROS_LOCALHOST_ONLY=1`을 사용합니다.
