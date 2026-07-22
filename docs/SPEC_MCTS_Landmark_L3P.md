# SPEC — MCTS-over-Landmarks bền vững với bất định (mở rộng L³P)

> **Mục đích tài liệu:** đủ chi tiết để một coding agent hiện thực hoá được toàn bộ hệ thống và chạy được bộ thí nghiệm E1, mà không cần đọc lại lịch sử trao đổi.
> **Trục nghiên cứu đã chốt:** Trục A — *uncertainty / robustness*. MCTS phải làm được điều Soft Floyd không làm: ra quyết định tốt hơn **khi ước lượng khoảng cách `V` bị nhiễu**.
> **Baseline gốc:** L³P (*World Model as a Graph* — latent landmarks + Soft Floyd + online planning, Algorithm 1). KHÔNG so với Hard Floyd.

---

## 0. Câu hỏi nghiên cứu và tiêu chí "thành công"

L³P đã có sẵn planner (Soft Floyd + Algorithm 1). Đóng góp của việc thêm MCTS **chỉ hợp lệ** nếu trả lời được:

> *Khi `V(cᵢ,cⱼ)` là ước lượng nhiễu, MCTS (explore/exploit qua UCB, sample-based rollout, uncertainty-aware) có ra quyết định robust hơn Soft Floyd (tin `V` một cách tất định) không?*

Câu này đánh trúng điểm yếu L³P tự thừa nhận: *"neural distance estimates are not entirely accurate"*. Nếu chứng minh được, insight chuyển thẳng sang bối cảnh VLA (world model estimate cũng nhiễu y hệt).

**Điều kiện đủ để tuyên bố thành công:** hai đường cong `success-rate vs σ_noise` tách nhau — MCTS degrade chậm hơn Soft Floyd khi `σ` tăng — VÀ tại `σ=0` hai đường TRÙNG nhau (sanity check).

---

## 1. Kiến trúc phân tầng (bắt buộc giữ đúng)

Hệ thống có 2 tầng thời gian. **MCTS chỉ hoạt động ở tầng cao (macro), chọn landmark. Việc đi giữa hai landmark giao cho low-level policy π đã học sẵn — KHÔNG plan chi tiết ở tầng dưới.** Đây là *temporal abstraction* của L³P, không phải thiếu sót.

```
┌────────────────────────────────────────────────────────────┐
│ HIGH-LEVEL (MCTS) — "đi tới landmark nào tiếp theo"          │
│   Không gian:      rời rạc, N landmark (+ goal)              │
│   Một macro-step = đi hết từ landmark này tới landmark kia    │
│   Transition model: V(cᵢ,cⱼ) — số bước ước tính (≈ world model)│
└────────────────────────────────────────────────────────────┘
                     ↓ giao subgoal = f_D(c_j)
┌────────────────────────────────────────────────────────────┐
│ LOW-LEVEL (policy π đã học qua HER) — "đi thế nào"           │
│   Không gian:      continuous action thật của env            │
│   Một bước = một env.step() thật                             │
│   Agent tự navigate K bước, KHÔNG cần MCTS ở tầng này         │
└────────────────────────────────────────────────────────────┘
```

**Lý do không plan chi tiết giữa hai landmark:** (1) nếu MCTS plan từng action thì quay lại đúng bài toán horizon-500 mà L³P né được; (2) latent space được huấn luyện sao cho khoảng cách phản ánh reachability, nên hai landmark liền kề là điểm π *tin cậy đi được* trong tầm ngắn (`d_max` cutoff của L³P đảm bảo điều này); (3) chính vì π tự đi nên `V` chỉ là *ước tính* — đây là nguồn uncertainty mà trục A nghiên cứu.

---

## 2. MCTS trên graph landmark (discrete MCTS)

Sau khi có `N` landmark, không gian search là **hữu hạn** — mỗi macro-step chọn 1 trong `N` landmark. UCT cổ điển áp dụng trực tiếp.

```
State của MCTS:   s_mcts = landmark hiện tại (hoặc state thật đã encode thành virtual node)
Action:           chọn một trong N landmark (hoặc goal) làm đích tiếp theo
Transition:       stochastic — vì V không chắc chắn (đây là điểm khác Floyd)
Terminal:         đạt goal, HOẶC vượt macro-horizon H_max
```

Bốn pha:

```
SELECTION (UCT):
  a* = argmax_j [ Q(node, LM_j)
                  + c · sqrt( ln N(node) / N(node, LM_j) )
                  + β · σ_V(node, LM_j) ]          # uncertainty bonus — xem §3

EXPANSION:
  Khi tới node chưa expand hết, thêm 1 landmark con chưa thử.

SIMULATION (rollout):
  Từ node mới, LIÊN TỤC chọn landmark theo Soft-Floyd-greedy (hoặc random)
  tới khi chạm goal hoặc hết rollout-horizon. Cộng dồn chi phí.
  → Ở đây V(·,·) đóng vai "dynamics model". Với NOISE Loại 2:
      V_sampled ~ N(V_obs, σ_V)   (sample MỖI lần dùng trong rollout)

BACKPROPAGATION:
  Backup tổng chi phí (hoặc thành/bại) lên các node trên đường đi.

CHỌN ACTION CUỐI: robust — kết hợp visit count N(root, c_j) và Q,
  KHÔNG chỉ max-Q (tránh exploit một nhánh may mắn nhờ noise).
```

**"World model" ở đây là hàm `V(g₁,g₂)`** — chỉ một forward pass MLP, cực rẻ, nên chạy được hàng nghìn simulation không lo latency. Đây là lý do thử nghiệm trong L³P-space sạch hơn VLA-space nhiều.

---

## 3. Reward function + đưa uncertainty vào MCTS

Reward cho một macro-step `cᵢ → cⱼ`:

```
R(cᵢ → cⱼ) =  − V(cᵢ, cⱼ)                    # (1) chi phí reachability (như Floyd)
             + λ_goal · 1[cⱼ = goal]          # (2) thưởng đạt đích
             − λ_risk · Uncertainty(cᵢ, cⱼ)   # (3) phạt bất định  ← MỚI, cốt lõi trục A
```

`Uncertainty(cᵢ,cⱼ) = var của V(cᵢ,cⱼ) estimate`. Nguồn của variance, tăng dần độ tinh vi:

```
Cách 1 (Phase 3 / E1, LÀM TRƯỚC): σ_V = σ_noise — biết trước vì CHÍNH TA inject.
        → "oracle uncertainty": MCTS được cho biết chính xác σ. Cố ý làm dễ
          để trả lời câu thuần: "NẾU biết uncertainty, MCTS có tận dụng được không?"
Cách 2 (sau): ensemble V-network, variance = độ bất đồng giữa các thành viên.
Cách 3 (nếu dùng contrastive/FOCA sau): spread của contrastive score làm proxy.
```

Hai chỗ có thể nhét uncertainty — **hiện thực CẢ HAI để ablate**:

- **Lựa chọn α — trong reward** (công thức (3) ở trên): đơn giản, phạt cạnh bất định. Nhược: risk-averse cứng, không phân biệt "bất định nhưng có thể tốt" với "bất định và tệ".
- **Lựa chọn β — trong exploration UCT** (số hạng `β·σ_V` ở §2): khuyến khích thử cạnh bất định để *giảm* bất định. Đúng chất MCTS hơn (Bayesian/UCB exploration) — chủ động giải quyết bất định qua simulation thay vì né mù quáng.

So sánh α vs β là một câu chuyện phụ đáng viết: *"nên coi uncertainty là risk-to-avoid hay information-to-gather?"*

---

## 4. [PHẦN (a) — MỚI THIẾT KẾ] Xử lý case (2): "agent đi nhưng không tới đúng landmark"

Sau khi MCTS chọn subgoal `c_j`, agent chạy π trong `K = round(−V(encode(s), c_j))` bước. Ba khả năng:

```
(1) REACHED               → tới đúng c_j (trong ~K bước)
(2) PROGRESSED_ELSEWHERE  → tới state thật s' ≠ c_j nhưng ĐÃ đi được quãng đáng kể
(3) STUCK                 → gần như không nhúc nhích / hết K bước chưa đi đâu
```

Dưới noise **Loại 2**, case (2) xảy ra **thường xuyên** → phải xử lý tử tế, nếu không success rate tụt oan và không phản ánh đúng năng lực MCTS. Cơ chế gồm 5 khối.

### 4.1. Phân loại kết quả sau macro-step

```python
def classify_outcome(s_start, s_end, subgoal_c, V, tau_reach, tau_progress):
    z_start = f_E(g(s_start))
    z_end   = f_E(g(s_end))
    dist_to_goal   = -V(z_end,   subgoal_c)     # còn cách đích bao xa
    dist_travelled = -V(z_start, z_end)         # đã đi được bao xa (theo latent metric)

    if dist_to_goal <= tau_reach:
        return "REACHED"
    elif dist_travelled >= tau_progress:
        return "PROGRESSED_ELSEWHERE"           # case (2)
    else:
        return "STUCK"                          # case (3)
```

> `tau_progress` là ranh giới giữa case (2) và (3) — **hyperparameter cần quét** (§8). Đặt quá thấp → mọi thứ thành case (2), không bao giờ blacklist; quá cao → phí tiến độ hợp lệ.

### 4.2. Recovery cho case (2) — snap-to-nearest rồi re-root

```python
def recover_case2(z_end, landmarks, V, tau_snap):
    # tìm landmark gần state thật nhất
    k = argmax_k  V(z_end, landmarks[k])        # = argmin khoảng cách
    if -V(z_end, landmarks[k]) <= tau_snap:
        return ("SNAP", k)                      # coi như đang ở landmark c_k → MCTS re-root tại c_k
    else:
        return ("VIRTUAL_ROOT", z_end)          # off-graph: dùng z_end làm root ảo
```

- **SNAP:** MCTS re-root tại `c_k` — tận dụng được tiến độ, không mất công đi lại.
- **VIRTUAL_ROOT (off-graph):** tính lại TOÀN BỘ cạnh từ vị trí thật `d_{z_end→c} = −V(z_end, c)` cho mọi landmark (đúng cơ chế re-encode của L³P), chạy lại MCTS từ `z_end`.

Cả hai nhánh: `Cnt_macro ← 0` để buộc re-plan macro ở vòng sau.

### 4.3. Execution feedback → hiệu chỉnh V (cầu nối sang Loại 1 / E1c)

Đây là kênh khiến MCTS **hơn** Soft Floyd ở việc phát hiện bias: Soft Floyd tính path một lần đầu episode, không sửa; MCTS học từ execution thật.

```python
# chi phí THỰC đã bỏ ra cho lần thử (cᵢ → cⱼ):
realized_cost = k_used + max(0, -V(z_end, c_j))      # số bước đã đi + phần dư còn lại

# cập nhật ước lượng theo execution (EMA), rồi feed ngược vào planner trong CÙNG episode:
V_exec[i][j] = (1 - ρ) * V_exec[i][j] + ρ * (-realized_cost)
V_obs[i][j]  = blend(V_prior[i][j], V_exec[i][j])    # planner dùng V_obs này ở macro-step sau
```

`ρ` = learning rate của EMA (hyperparameter). `blend` có thể là trung bình có trọng số theo số lần đã traverse cạnh đó.

### 4.4. Failure budget / loop guard

```python
attempts[c_j] += 1
if attempts[c_j] >= R_max:                # thử R_max lần vẫn không REACHED
    d_macro[c_j] = -inf                   # blacklist c_j cho phần còn lại của episode (như L³P)
if total_env_steps >= H_max:
    return FAIL_EPISODE                   # chặn vòng lặp vô hạn
```

Case (3) (STUCK) áp dụng blacklist **ngay** (không tin việc đã đi); case (2) chỉ blacklist sau `R_max` lần lặp — khác biệt then chốt giữa hai case.

### 4.5. Nối với uncertainty bonus

Mỗi case-(2)/(3) lặp trên một cạnh làm tăng variance thực nghiệm `σ_V` của cạnh đó (đo từ chênh `realized_cost` vs `V_obs`). `σ_V` cao → `β·σ_V` (Lựa chọn β) hoặc `λ_risk·Unc` (Lựa chọn α) tăng → MCTS **tự né** cạnh hay lệch mà không cần thêm luật cứng. Đây là điểm hợp nhất cơ chế (a) với thiết kế trục A.

---

## 5. Mô hình noise (Loại 1 / Loại 2) và ba vị trí inject (Nơi 1/2/3)

### 5.1. Hai loại noise — bản chất khác nhau, test khả năng khác nhau

**Loại 1 — Estimation bias (cố định mỗi edge, sample MỘT lần / episode):**
```
V_obs(cᵢ,cⱼ) = V_true(cᵢ,cⱼ) · (1 + b_ij),   b_ij ~ N(0, σ²)   (sample 1 lần)
```
Mô phỏng world model học SAI một số edge có hệ thống ("wormhole", L³P Fig.6). Env thật KHÔNG nhiễu — chỉ *estimate* của planner sai. Test khả năng MCTS **phát hiện estimate sai qua execution** (dùng §4.3) và điều chỉnh.

**Loại 2 — Execution stochasticity (sample MỖI lần đi):**
```
V_actual(cᵢ,cⱼ) = V_true(cᵢ,cⱼ) + η,   η ~ N(0, σ²)   (sample mỗi lần traverse)
```
Mô phỏng π không tất định — cùng cặp landmark, lần này 20 bước, lần sau 25. Test khả năng MCTS **lập luận về kỳ vọng dưới stochastic transition** (sample-based rollout). **LÀM TRƯỚC** vì khớp trực tiếp thiết kế rollout sample-based; Loại 1 sau (cần cơ chế "học từ execution" ở §4.3).

|                       | Loại 1 (bias cố định)               | Loại 2 (stochastic)                    |
|-----------------------|-------------------------------------|----------------------------------------|
| MCTS thắng nhờ        | Execution feedback → phát hiện bias | Sample rollout → ước tính kỳ vọng đúng |
| Soft Floyd thua vì    | Tin edge bias một lần, không sửa    | Chỉ dùng điểm ước tính, bỏ qua variance|
| Cơ chế cần trong MCTS | Cập nhật V từ execution (§4.3)      | Sample V trong rollout (§2)            |

### 5.2. Ba nơi `V` xuất hiện — noise chạm vào đâu quyết định thí nghiệm có ý nghĩa

```
Nơi 1 — Planner nhìn thấy (input cho MCTS/Floyd):   V_obs    → chọn đường
Nơi 2 — Rollout simulation bên trong MCTS:          V_sampled → đánh giá đường
Nơi 3 — Môi trường THẬT (agent thực sự đi mất bao nhiêu bước): V_actual → quyết thành/bại
```

**Thí nghiệm đầu tiên (sạch nhất): noise CHỈ ở Nơi 1&2, giữ Nơi 3 = V_true.** Planner bị "lừa" bởi estimate nhiễu, nhưng agent thật đi trong env đúng; metric success đo trên `V_true`. Câu chuyện: *"MCTS ra quyết định tốt hơn Soft Floyd KHI cả hai nhìn cùng một estimate nhiễu."* Đảm bảo **so sánh công bằng**: cả MCTS và Floyd nhận cùng `V_obs`, khác biệt duy nhất là *cách xử lý*.

Noise vào Nơi 3 (env cũng stochastic) khó hơn, thực tế hơn — để E1d.

---

## 6. Vòng lặp chính (pseudocode đầy đủ)

```python
s = env.reset()
Cnt_macro = 0
attempts = defaultdict(int)
total_env_steps = 0

while not done and total_env_steps < H_max:
    if Cnt_macro <= 0:
        subgoal_c = MCTS_Landmark(s, goal, V_obs,      # §2, §3
                                  sigma_V=sigma_noise,  # oracle uncertainty (Cách 1)
                                  d_macro=d_macro)      # blacklist đã áp dụng
        K = round(-V_obs(f_E(g(s)), subgoal_c))
        Cnt_macro = K
        s_start = s
        k_used = 0

    # LOW-LEVEL: agent tự đi, KHÔNG plan chi tiết
    a = pi(s, f_D(subgoal_c))
    s = env.step(a)                     # Nơi 3: V_actual (Loại 2 nếu inject env)
    total_env_steps += 1
    k_used += 1
    Cnt_macro -= 1

    reached = reached_subgoal(s, subgoal_c, tau_reach)
    if reached or Cnt_macro <= 0:
        outcome = classify_outcome(s_start, s, subgoal_c, V_obs, tau_reach, tau_progress)  # §4.1
        update_V_exec(s_start, s, subgoal_c, k_used, V_obs)                                # §4.3

        if outcome == "REACHED":
            attempts.clear()                        # reset khi có tiến triển thật
        elif outcome == "PROGRESSED_ELSEWHERE":     # case (2) — §4.2
            mode, target = recover_case2(f_E(g(s)), landmarks, V_obs, tau_snap)
            attempts[subgoal_c] += 1
            if attempts[subgoal_c] >= R_max:
                d_macro[subgoal_c] = -inf           # §4.4
        else:  # STUCK — case (3)
            d_macro[subgoal_c] = -inf               # blacklist NGAY
        Cnt_macro = 0                               # buộc re-plan macro
```

---

## 7. Baselines bắt buộc so sánh (tách biến)

```
Baseline 1: Soft Floyd (L³P gốc)                 — shortest path TĨNH, softmax relaxation
Baseline 2: Naive re-plan mỗi bước (L³P Fig.8)   — để tách "MCTS thắng nhờ re-plan thường xuyên?"
Baseline 3: Soft Floyd + re-plan mỗi bước với V_obs mới  — tách rõ hơn nữa biến "re-plan"
Của ta:     MCTS-over-landmarks (full: rollout + uncertainty bonus)
Ablation:   MCTS-β=0 (tắt uncertainty bonus, VẪN sample rollout)  ← quan trọng nhất, xem E1b
```

**Cảnh báo:** Soft Floyd của L³P **đã** dùng softmax relaxation để robust với "occasional bad edges" (L³P §5.4) → KHÔNG phải baseline ngây thơ. Phải so với SOFT Floyd (đã tune, `d_max` rất nhạy — tune công bằng), tuyệt đối không so Hard Floyd.

---

## 8. Ma trận thí nghiệm E1 + sanity check + expected curves

| Thí nghiệm | Noise loại | Nơi inject | σ quét            | Baseline so sánh          | Câu hỏi trả lời                              |
|------------|-----------|------------|-------------------|---------------------------|----------------------------------------------|
| **E1a**    | Loại 2    | Nơi 1&2    | {0, 0.1, 0.3, 0.5}| Soft Floyd, Naive         | MCTS + sample-rollout có robust hơn?         |
| **E1b**    | Loại 2    | Nơi 1&2    | {0, 0.1, 0.3, 0.5}| MCTS-β=0                  | Uncertainty bonus có đóng góp RIÊNG?          |
| **E1c**    | Loại 1    | Nơi 1&2    | {0, 0.1, 0.3}     | Soft Floyd                | MCTS phát hiện bias qua execution (§4.3)?     |
| **E1d**    | Loại 2    | Nơi 1,2,**3** | {0, 0.3}       | Soft Floyd                | Còn thắng khi env cũng nhiễu?                 |

**E1b là thí nghiệm quan trọng nhất về mặt khoa học** — tách "MCTS thắng nhờ *tree search/lookahead*" hay nhờ *cơ chế uncertainty*. Nếu MCTS-β=0 đã thắng Floyd, đóng góp là "lookahead" chứ không phải "uncertainty" → viết claim cho đúng.

**Đường cong kỳ vọng:**
```
Success rate
1.0 ┤ ●━━━●   ← σ=0: hai đường TRÙNG (SANITY CHECK bắt buộc)
    │      ╲╲
    │       ╲ ╲___  MCTS (degrade chậm)
    │        ╲    ╲___
    │  Soft Floyd ╲____╲  (degrade nhanh)
    └────┬────┬────┬────┬──→ σ_noise
        0   0.1  0.3  0.5
```

**Ba kịch bản đọc kết quả:**
```
KB1 — Hai đường tách rõ:        ✓ giả thuyết đúng, viết được bài
KB2 — Hai đường trùng nhau:     Soft Floyd đã đủ robust → cần σ mạnh hơn HOẶC đổi Knob (adversarial)
KB3 — MCTS thua ngay cả σ=0:    CÓ BUG (lookahead không nên hại) → debug trước, đừng làm gì tiếp
```

**Sanity check bắt buộc (làm trước E1):** tại `σ=0`, MCTS phải ≈ Soft Floyd. Nếu khác → implementation hoặc hyperparameter không công bằng → SỬA trước khi chạy sweep.

---

## 9. Cấu trúc code đề xuất

```
project/
  planners/
    soft_floyd.py        # baseline gốc (giữ nguyên từ L³P, chỉ thêm hook nhận V_obs nhiễu)
    mcts_landmark.py     # §2 selection/expansion/simulation/backprop + UCT có β·σ_V
    reward.py            # §3 R(cᵢ→cⱼ), Lựa chọn α/β (config-flag)
  execution/
    macro_loop.py        # §6 vòng lặp chính
    recovery.py          # §4 classify_outcome, recover_case2, update_V_exec, loop guard
  noise/
    injector.py          # §5 Loại 1/Loại 2 × Nơi 1/2/3, bật/tắt qua config
  eval/
    experiment_e1.py     # §8 chạy E1a-d, log success-rate vs σ, vẽ đường cong
    sanity.py            # §8 σ=0 → MCTS≈Floyd
  config.yaml
```

**Config knobs cần expose (không hardcode):**
```yaml
mcts:      { c_uct, beta, n_simulations, rollout_horizon, H_max }
reward:    { lambda_goal, lambda_risk, uncertainty_mode: [alpha|beta|both] }
recovery:  { tau_reach, tau_progress, tau_snap, R_max, rho_ema }   # §4 — quét tau_progress
noise:     { type: [1|2], locations: [1,2] | [1,2,3], sigma }
floyd:     { d_max }        # tune công bằng, nhạy
env:       { name: AntMaze-Hard, n_landmarks: 50 }
```

**Tái dùng từ L³P (KHÔNG viết lại):** encoder/decoder `f_E/f_D`, hàm `V`, low-level policy `π` (HER), Soft Floyd, cơ chế `Cnt` + blacklist trong Algorithm 1. Phần MỚI chỉ là: `mcts_landmark.py`, `reward.py` (uncertainty), `recovery.py` (§4), `noise/injector.py`.

---

## 10. Acceptance criteria & rủi ro

**Acceptance (theo thứ tự, dừng nếu fail):**
1. Sanity: tại `σ=0`, `|success_MCTS − success_Floyd| < ε_tol`. FAIL → có bug, dừng.
2. E1a: MCTS degrade chậm hơn Floyd khi `σ` tăng (KB1).
3. E1b: MCTS-full > MCTS-β=0 → xác nhận uncertainty bonus có đóng góp riêng.
4. E1c: có execution feedback (§4.3), MCTS phát hiện được bias (thắng Floyd ở Loại 1).

**Ba rủi ro lớn (lường trước):**
- **R1 — MCTS không thắng Floyd dù có noise.** Soft Floyd (softmax) có thể đã đủ robust. → test E1a SỚM; nếu KB2 thì tăng `σ` hoặc chuyển Knob adversarial.
- **R2 — Graph quá nhỏ để lookahead có ý nghĩa.** N=50 + env ngắn → shortest path 2-3 landmark, MCTS không tạo khác biệt. → dùng **AntMaze-Hard (horizon ~500)**.
- **R3 — Confound "MCTS tốt hơn" vs "tune tốt hơn".** Phải tune Soft Floyd công bằng (`d_max` nhạy), nếu không reviewer nghi ngờ.

**Chỉ số phải log mỗi episode:** success (đo trên `V_true`), số macro-step, số lần case(2)/case(3), số cạnh bị blacklist, latency MCTS. Báo cáo mean ± CI (bootstrap) qua nhiều seed.
