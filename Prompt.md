"Work on one milestone at a time. After finishing, stop and wait for me to say START for the next one."

ROLE
You are my senior ML engineer, full-stack mentor, and project manager. I am a final-year CS student with working knowledge of Python, React, TypeScript, and FastAPI. I have NO local GPU, so all training happens on Google Colab (free T4 GPU, 12h session limit, sessions disconnect). I want to finish this university assignment in the LEAST total time with the highest grade. Give me complete, copy-paste-ready, tested-style code, not outlines.

HOW YOU MUST WORK
1. We work in 4 milestones (below). Do ONE milestone at a time. Do not move on until I type "MILESTONE X DONE" and paste any errors/results.
2. For each milestone, give me in this order:
   a) Goal and definition of done
   b) Exact folder/repo structure additions
   c) Complete code files (full content, not snippets), with Colab cells in order
   d) Exact commands to run, expected runtime on a T4, and expected output
   e) "Evidence checklist": every screenshot, plot, table, and number I must save for the IEEE report
   f) Common errors and fixes
   g) Short "viva notes": 5-8 questions an evaluator could ask about this milestone, with answers
3. Every design choice (architecture, loss, hyperparameters, bottleneck size, skip connections, gate temperature, balance loss, etc.) must come with a short justification and 1-2 real references (paper/doc names) so I can cite them in the "research-informed decisions" part of the report. Never invent references; if unsure, say so.
4. Optimise for time: small but valid models, cached data, few-epoch Optuna trials with pruning, resumable checkpoints saved to Google Drive, and Optuna storage in SQLite on Drive so a disconnect doesn't lose studies.
5. Keep anything I must do by hand (Google Stitch design, screen recording, YouTube upload, LaTeX compile) clearly marked "MANUAL STEP" with exact instructions.
6. Warn me about anything in the spec that is easy to get wrong.

ASSIGNMENT SPEC (authoritative)
Overview: Four generative-AI systems integrated into ONE browser app. Individual work. AI tools allowed but I must verify and understand everything; evaluator may ask me to modify code or test unseen images live.

Datasets
- Tasks 1-3: Oxford-IIIT Pet. Use official trainval as development data; split 80/20 train/val with random seed 42. Official test set untouched until final evaluation. Convert to RGB, resize to 128x128. Same split for Tasks 1, 2, 3. Class labels not needed; originals are clean targets.
- Task 4: FS2K (2,104 photo-sketch pairs, 3 sketch styles). Use official train/test definition. Hold out 15% of official train as validation, stratified by sketch style, seed 42. Resize both to 128x128, keep pairing. Test set not used for training or hyperparameter selection. Spatial augmentation must be identical for photo and sketch.

Runtime corruption (training): applied inside the data-loading pipeline, new type/severity sampled each time an image is loaded, never saved to disk. Each training image picks one of 4 conditions with equal probability:
- Clean: unchanged
- Salt-and-pepper: p ~ U(0.02, 0.15); replaced pixels black or white 50/50
- Gaussian blur: kernel in {3,5,7}, sigma ~ U(0.5, 2.5)
- Rectangular occlusion: 1-3 black rectangles, jointly covering 10-35% of image area, random locations
The corruption label is recorded (used for the classifier).
Validation and test corruptions: deterministic. Generate a validation manifest and a test manifest ONCE, storing corruption type, severity, mask coordinates, blur settings, and seed per image.
Final test severities (3 fixed levels per corruption): S&P p = 0.03, 0.08, 0.15; blur (kernel, sigma) = (3,0.7), (5,1.5), (7,2.5); occlusion about 10%, 20%, 35% area using 1, 2, 3 rectangles.

Task 1: Universal denoising autoencoder
- One model, not told the corruption type. Conv encoder (spatial dims down, channels up), genuine compressed latent bottleneck, conv decoder to 128x128 RGB. No unrestricted skip connections; if limited skips are used, investigate and justify them in the report (do an ablation).
- Loss: L = alpha * L1(x, x_hat) + (1 - alpha) * (1 - SSIM(x, x_hat)); start alpha = 0.8 but final value chosen by Optuna.
- Optuna: at least learning rate, batch size, bottleneck dimension, encoder channels, dropout, alpha. Objective combines reconstruction quality and SSIM. Report the search space, number of completed trials, best trial, and final config.
- Results: separately for clean, S&P, blur, occlusion, AND by low/medium/high severity. Visuals: clean target, corrupted input, reconstruction, absolute error map. At least 12 representative examples and 4 discussed failure cases.
- Export to ONNX. App workspace name: "Universal Restoration" (upload corrupted image or pick clean sample and apply a corruption; show input, restored output, corruption settings, inference time).

Task 2: Hard-routed specialists
- CNN classifier with 4 classes (clean, salt, blur, occlusion), labels from the runtime pipeline, class-balanced batches, cross-entropy. Optuna: learning rate, batch size, channel configuration, dropout, weight decay. Report accuracy, macro precision/recall/F1, per-class metrics, normalized 4x4 confusion matrix.
- Three specialist autoencoders (salt, blur, occlusion), each trained only on its corruption, independently trained parameters, same clean target. Allowed: one shared Optuna search for a common architecture, then train the three independently. Tune learning rate, bottleneck size, channel config, batch size, L1/SSIM weight.
- Inference: argmax of classifier probabilities selects the specialist; clean input uses an IDENTITY BYPASS.
- Evaluate in two modes: oracle routing (manifest label) and predicted routing (classifier). Identify and discuss cases where classifier errors cause restoration failure.
- App workspace name: "Hard-Routed Restoration" (show 4 probabilities, predicted class, selected expert, output, inference time). Export classifier and all 3 specialists to ONNX.

Task 3: Soft mixture-of-experts
- Gate network + 3 specialist experts from Task 2 + identity branch. Weights w = softmax(g(x)/T) over [identity, salt, blur, occlusion]. Output = w0*x + w1*E_salt(x) + w2*E_blur(x) + w3*E_occ(x). Differentiable, trained jointly.
- Initialise gate from the Task 2 classifier and experts from the Task 2 specialists (no random start). Warm-up: freeze experts, train only the gate; then unfreeze and jointly fine-tune with a smaller learning rate.
- Loss: L = l_rec*L1 + l_s*(1-SSIM) + l_c*CE(gate, known label) + l_b*L_balance. Initial values: 0.8, 0.2, 0.1, 0.01. Balance loss example: sum over i of (mean weight_i - 1/4)^2 on a balanced batch; alternatives allowed if researched and justified.
- Optuna: joint fine-tuning learning rate, temperature T, classification weight, balance weight, reconstruction weighting. Pruning allowed (e.g., on routing collapse).
- Analysis: average expert weights per true corruption type and per severity; examples where one expert dominates and where weights are spread; routing heatmap/weight-distribution diagram; check for inactive experts or one expert dominating unrelated inputs.
- App workspace name: "Soft Mixture-of-Experts Restoration" (show all 4 weights, output, inference time, visual indication of which experts contributed). Export the COMPLETE soft-MoE pipeline to ONNX.

Task 4: Style-conditioned face-to-sketch cGAN
- Generator: U-Net style, input = photo + style condition (3 categories). Discriminator: PatchGAN on (photo, style, sketch). Style via learned categorical embedding, used INSIDE both generator and discriminator (not just a UI label).
- Generator loss: L_G = L_adv + lambda_L1 * L1(y, G(x,s)); start lambda = 100 but tune with Optuna. BCE-with-logits for adversarial parts.
- Optuna: generator LR, discriminator LR, batch size, base channels, dropout, style-embedding dimension, lambda_L1. Optuna trials may use fewer epochs; then retrain the best config for the full schedule.
- Augmentation: identical spatial transforms applied to both photo and sketch.
- Log separately: D real loss, D fake loss, G adversarial loss, G reconstruction loss, validation metrics. Log samples at fixed intervals using the SAME validation photos.
- App workspace name: "Face-to-Sketch Generator" (upload or webcam, choose Style 1/2/3, side-by-side original/sketch, download button). Export generator only to ONNX.

Tooling requirements
- PyTorch (or TF). Optuna in ALL four tasks. MLflow or Weights & Biases for experiments, hyperparameters, losses, results, checkpoints, visual outputs.
- ONNX export for all inference models + a script verifying ONNX output matches PyTorch within tolerance.
- Frontend: React + Tailwind CSS. Backend: FastAPI. UI must first be designed in Google Stitch (evidence screenshots in report). One app with 4 workspaces. Backend endpoints at minimum: health check, universal restoration, hard routing, soft mixture, face-to-sketch; validate uploads; preprocess; run ONNX (onnxruntime); return results + routing + timing.
- Docker: frontend and backend in containers; docker-compose starts everything with one documented command. Evaluator will clone the repo, download model files, run compose, open the browser.
- GitHub repo containing: source, configs, requirements, data-prep scripts, training scripts, evaluation scripts, Optuna studies, ONNX export code, app code, Dockerfiles, docker-compose, README with full instructions. Do NOT commit full datasets or huge model files (use Git LFS or a documented download link).
- Report: IEEE format via LaTeX, research-paper style: intro, related work, dataset prep, architecture, losses, training procedure, Optuna design, experimental setup, results, analysis, application architecture, limitations, conclusion, with separate sections per task, repo link, YouTube link, AI-use appendix. Must include architecture diagrams, corruption config, training/validation curves, Optuna results, confusion matrix, quantitative tables, routing-weight visualisations, image grids, error maps, app screenshots, failure cases. Every figure/table must be interpreted in text. 
- Demo video 5-7 min on YouTube (link only in report) showing: startup, image upload, runtime corruption, universal restoration, hard routing, soft expert weights, face-to-sketch, result download, and the experiment-tracking records.
- AI-use appendix: tools used, for which tasks, how outputs were tested/corrected.

MILESTONE PLAN (follow this)

MILESTONE 1: Foundations + Task 1 + Task 2 (Colab)
- Repo skeleton, Colab setup cell (mount Drive, install deps, fix seeds), tracking setup (choose W&B or MLflow and justify; prefer what is fastest for screenshots).
- Download Oxford-IIIT Pet; build the 80/20 split with seed 42; cache resized 128x128 RGB uint8 arrays on Drive for fast loading (Colab has few CPU cores; make corruption fast, e.g., OpenCV/torch ops, and explain how it stays inside the data-loading pipeline).
- Implement corruption functions exactly as specified; generate and save validation and test manifests (JSON/CSV) with all required fields; build a deterministic fixed-severity test set; unit tests proving determinism.
- Task 1: model, L1+SSIM loss, Optuna study (small trial count with pruning, sensible epochs per trial; tell me realistic numbers for a T4), final training, per-corruption and per-severity evaluation, 12 example grid with error maps, 4 failure cases, skip-connection ablation if I use skips, ONNX export + PyTorch-vs-ONNX check.
- Task 2: balanced-batch classifier + Optuna + metrics + confusion matrix; shared-architecture Optuna for specialists, then train 3 specialists independently; oracle vs predicted routing evaluation; failure analysis from classifier errors; ONNX export of the 4 models + verification.
- Save all checkpoints and ONNX files to Drive. Provide the evidence checklist.

MILESTONE 2: Task 3 + Task 4 (Colab)
- Task 3: gate from classifier, experts from specialists, identity branch, warm-up then joint fine-tuning, joint loss with balance term (justify with research), Optuna with pruning on routing collapse, evaluation, per-corruption and per-severity average weights, heatmap, dominating/spread examples, inactive-expert check, full-pipeline ONNX export (gate + experts + weighted sum in one graph, also outputting the weights) + verification.
- Task 4: FS2K download/parsing (tell me exactly where to get it and how the official train/test split and style labels are stored; warn me if the dataset needs manual download or access request and give a fallback plan), stratified 15% validation split seed 42, paired augmentation, conditional U-Net generator, conditional PatchGAN discriminator with embeddings used inside both, separate loss logging, fixed validation samples logged at intervals, Optuna with short trials, then full retraining of the best config, validation metrics (e.g., L1, SSIM, optionally LPIPS/FID if cheap), generator-only ONNX export + verification.
- Evidence checklist for both tasks.

MILESTONE 3: Application + Docker + Stitch (local machine)
- MANUAL STEP: give me the exact Google Stitch prompt(s) to generate the UI for 4 workspaces (Universal Restoration, Hard-Routed Restoration, Soft Mixture-of-Experts Restoration, Face-to-Sketch Generator), including layout, colours, cards, image panels, and responsive behaviour, and tell me which screenshots to save as evidence.
- FastAPI backend: file validation, preprocessing, onnxruntime sessions loaded once at startup, runtime corruption endpoint (so the user can pick corruption type and severity), endpoints for health, universal, hard-route, soft-mix, face-to-sketch; timing; base64 image responses; CORS; error handling.
- React + Tailwind frontend with the four workspaces: upload/sample selection, corruption controls, probability bars, routing weight bars/heatmap with expert contribution indication, inference time, webcam capture, style selector, side-by-side view, download button.
- Dockerfiles for both, docker-compose.yml, model files mounted from a ./models folder with a documented download script/link, one-command startup, README instructions.
- Test checklist for the live evaluation (unseen images, already-corrupted uploads, restart from a clean clone).

MILESTONE 4: Report, repo, video (mostly manual)
- Finalise repo structure and README; list what goes into Git LFS or download links; where the Optuna studies and experiment-tracking exports live.
- Provide a complete IEEE LaTeX project (Overleaf-ready: main.tex, references.bib, section files, figure folders, table templates) with all sections pre-written as drafts that I fill with my real numbers and figures; separate sections per task; AI-use appendix draft; limitations and conclusion.
- Provide a figure/table checklist mapped to the rubric requirements above.
- Provide a minute-by-minute script for the 5-7 minute demo video covering every required item.
- Provide a final submission checklist (GitHub link, YouTube link, ONNX links, report PDF, deadline) and a viva-prep sheet with the 30 most likely questions and answers.

TIME-SAVING DEFAULTS YOU SHOULD ASSUME
- Small, fast models (e.g., 4 conv stages, base channels 32-64), mixed precision (AMP), AdamW, cached data on Drive, checkpoint every epoch, resume logic, Optuna with SQLite storage + MedianPruner, small trial counts (tell me the minimum defensible numbers), W&B or MLflow logging every trial.
- Reuse one shared training utilities module across tasks.
- Prefer ONNX-friendly ops (avoid ops that break export), opset 17.

START NOW
First, reply with: (1) a one-page summary of the assignment as you understand it, (2) anything ambiguous or risky in the spec and your proposed safe interpretation, (3) a realistic Colab GPU-hours budget per milestone, and (4) the exact repository folder structure. Then wait for me to say "START MILESTONE 1".