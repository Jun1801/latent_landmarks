# Roadmap: test-time planning (MCTS-over-landmarks) → VLA

> **Tầm nhìn:** tăng năng lực của một base policy (VLA) **đóng băng** bằng
> **test-time planning** trên một **world-model đã học**, với planner **robust
> khi world-model ước lượng SAI**. Base scaffold = L³P. Kiểm chứng trong sim
> trước, robot thật sau cùng.
> **Mục tiêu:** paper/thesis. **Nguồn lực:** GPU/Kaggle (không robot thật ngay).

## 0. Kết quả đã có (trên NumPy PointMaze, 1 checkpoint)
- **E1c (positive):** V có **bias hệ thống** → MCTS + execution-feedback phát hiện
  & sửa, thắng Soft-Floyd tĩnh (0.87 vs 0.52), robust qua τ. Con âm trước đây (0.43)
  là artifact của bug goal-blacklist (đã fix, `_candidate_mask`).
- **E1a (negative):** nhiễu 0-mean → **fresh-graph-replan (MPC) 0.97 > MCTS 0.65**.
  Tree-search KHÔNG thắng replan đơn giản khi nhiễu 0-mean.
- **E1b/E1d:** yếu/chưa kết luận (β-bonus ~vô hiệu; E1d gần trần nên chưa tách).
- Mọi checkpoint MuJoCo (pointmaze_mujoco, AntMaze) **undertrained** → không dùng được
  (AntMaze: kiến chưa học đi; pointmaze_mujoco: V median~33, graph không giúp).

## 1. Ánh xạ L³P ↔ VLA và chỗ GÃY
| L³P | VLA analog | Trung thực |
|---|---|---|
| `V(g1,g2)` MLP | world-model VLA ước lượng reachability | ✅ cả hai nhiễu/bias |
| π(s,g) | VLA action policy (frozen) | ✅ |
| latent landmark + soft-Floyd | subgoal/skill graph | ⚠️ **gãy #1: landmark là gì trong VLA?** |
| MCTS test-time | test-time planning | ✅ |
| execution feedback | thử-rồi-sửa trên robot | ⚠️ đắt/nguy hiểm ở thật |

**Chỗ gãy phải xử lý:** (#1) landmark trong không gian VLA cao chiều; (#2) chi phí
query world-model lớn (MCTS cần hàng nghìn query rẻ); (#3) execution feedback đắt/
nguy hiểm ở robot thật; (#4) bias thật có cấu trúc (OOD), khác Gaussian tiêm tay.

**Gỡ #1 (then chốt):** trong VLA, **landmark = tập skill/subgoal RỜI RẠC** (ngôn ngữ
hoặc skill library), không rải trong latent liên tục. → MCTS trên graph skill hữu
hạn, né hoàn toàn bài toán metric latent cao chiều.

## 2. VLA-sim faithful — 5 thành phần
1. **Task** long-horizon đa-bước có ngôn ngữ: LIBERO / CALVIN / ManiSkill2.
2. **Base policy (VLA đóng băng):** skill/goal-conditioned policy, hoặc VLA thật (Octo/OpenVLA).
3. **World-model học được** trả `V(subgoal_i, subgoal_j)` (kiểu TD-MPC/Dreamer) — *khó nhất*.
4. **Landmark = skill rời rạc.**
5. **Bias TỰ NHIÊN** của world-model (đo vs sim ground-truth), không tiêm tay.

### 3 mức faithfulness ↔ công sức
- **Tier 1 (cầu nối):** goal-conditioned + V học được trên env cao-chiều/dài-hạn hơn
  (AntMaze/ManiSkill) + **natural bias**. = P2 + P3. Vài ngày–1 tuần.
- **Tier 2 (đề xuất cho paper):** LIBERO/CALVIN + skill-conditioned policy + world-model
  skill-transition + landmark=skill + natural bias. **2–4 tuần.**
- **Tier 3 (future):** VLA thật (OpenVLA/Octo) + CALVIN. Tháng+.

## 3. Lộ trình
```
NGAY:   P1  — đa-seed PointMaze, "MCTS có thực sự giúp?" (+ ablation planner, xem §5)
KẾ:     Tier 1 = P2 (1 env khó hoạt động) + P3 (natural bias)  → "beyond toy"
ĐÍCH:   Tier 2 (VLA-sim skill-based)                          → chương "VLA-sim"
FUTURE: Tier 3 (VLA thật) + robot thật                        → roadmap/future work
```

## 4. Tái dùng vs xây mới
- **Tái dùng (đã test):** `LandmarkMCTS`, execution-feedback (§4), noise/bias harness,
  run-e1x + hierarchical CI, calibrate d_max.
- **Xây mới cho VLA-sim:** adapter benchmark→GoalEnv; skill/subgoal extractor (landmark
  rời rạc); world-model học `V(skill_i,skill_j)`; đo natural-bias vs ground-truth.

## 5. ⚠️ Design review MCTS (làm TRƯỚC khi scale) — xem chi tiết trong chat/MCTS.md
**Câu hỏi lớn:** trên graph landmark RỜI RẠC nhỏ (N≈50) đã có soft-Floyd giải all-pairs
shortest-path, MCTS có earn its keep không?
- **Bằng chứng nghi ngờ:** E1a MCTS < MPC-replan; E1b β-bonus vô hiệu; E1c **MCTS-nofb
  TỆ NHẤT** (win đến từ *feedback*, không phải tree-search). → *Chưa thí nghiệm nào cho
  thấy bản thân tree-search tạo ra chiến thắng.*
- **Hệ quả:** framing paper nên là *"uncertainty-aware test-time planning"* (thành phần
  ăn tiền = replan + risk-aversion + execution-feedback), với MCTS là **một** lựa chọn
  so cùng **MPC-replan** và **robust-shortest-path (non-MCTS)**.
- **MCTS chỉ thực sự cần** khi skill-space LỚN/continuous, không enumerate/giải-exact
  được → đó là regime **VLA (Tier 2/3)**, không phải toy N=50.
- **P1 nên gồm ablation planner** (static Floyd / MPC-replan / robust-shortest-path /
  MCTS-nofb / MCTS-fb) để trả lời thẳng "thành phần nào mới thực sự giúp".
