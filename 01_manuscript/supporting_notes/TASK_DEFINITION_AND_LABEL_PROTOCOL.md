# Task Definition And Label Protocol

## 1. Purpose

This document is now the canonical definition of the first formal paper task.

It locks:

1. the official input unit
2. the official output targets
3. the priority of the tasks
4. the label scope for each target
5. the split and runtime display policy

## 2. Data Hierarchy

### 2.1 Group Level

One group equals one experimental condition:

- one nodule size
- one nodule depth

Current matrix:

- 7 size conditions
- 6 depth conditions
- total `42` groups

### 2.2 Repetition Level

Each group has 3 repetitions:

- `1.CSV`
- `2.CSV`
- `3.CSV`

### 2.3 Frame Level

Each frame is one tactile stress map:

- one `12 x 8` grid

### 2.4 Segment Level

Each positive annotation is a contiguous frame segment where the nodule response is judged detectable.

### 2.5 Window Level

The neural network input unit is a sliding temporal window.

The formal first-version setting is:

- `T = 10`
- `stride = 2`

## 3. Formal Input Definition

One training and inference sample is defined as:

> a length-10 consecutive sequence of tactile stress maps extracted from one repetition of one size-depth group

Formal tensor shape:

- `X in R^(10 x 1 x 12 x 8)`

This definition is locked for the first paper version.

The following are not part of the first formal protocol:

- single-frame input
- `T = 16` long-window primary protocol

## 4. Official Output Targets

The first formal model version has 4 outputs.

### 4.1 Detection Output

- name: `y_det_prob`
- meaning: probability that a nodule-related response is present in the current window
- type: binary detection probability

This is the primary task.

### 4.2 Size Classification Output

- name: `y_size_cls`
- meaning: one of 7 size classes
- type: categorical classification

### 4.3 Size Regression Output

- name: `y_size_reg`
- meaning: continuous size estimate in `cm`
- type: regression

This runs in parallel with size classification.

### 4.4 Coarse Depth Output

- name: `y_depth_cls_coarse`
- meaning: one of 3 coarse depth classes
- type: categorical classification

Formal coarse depth groups:

- `shallow = {0.5, 1.0}`
- `middle = {1.5, 2.0}`
- `deep = {2.5, 3.0}`

The `6`-class fine depth target is not part of the first formal output contract.
It remains a later upgrade or supplementary experiment.

## 5. Official Task Priority

### 5.1 Primary Task

`Task A: Nodule presence detection`

Goal:

- stably determine whether a hidden nodule response is present during pressing

### 5.2 Main Inversion Task

`Task B: Nodule size inversion`

Goal:

- estimate the most likely size after detection becomes positive

### 5.3 Secondary Task

`Task C: Coarse depth training and interpretation`

Goal:

- train a coarse depth output
- use it together with non-model statistics to explain depth-related behavior

Depth is therefore part of the first formal model, but it is not the strongest paper claim.

## 6. Label Scope

### 6.1 Detection Label

Detection uses the formal window detection label from the paper protocol.

For the current paper path, this remains:

- `y_det = 1` if the formal detection rule marks the window as positive
- `y_det = 0` otherwise

Detection supervision applies to:

- all windows

### 6.2 Size Labels

Size labels apply to:

- positive windows
- or detection-gated positive candidate windows

They do not supervise obvious negative windows.

### 6.3 Depth Labels

Coarse depth labels apply to:

- positive windows
- or detection-gated positive candidate windows

They do not supervise obvious negative windows.

## 7. Official Training Strategy

The first formal training organization is two-stage.

### Stage 1

Train:

- encoder + detection head

Goal:

- stable nodule probability output

### Stage 2

Train:

- size classification head
- size regression head
- coarse depth classification head

Using:

- positive windows
- or detection-gated positive windows

Goal:

- learn inversion behavior without contaminating the inversion heads with meaningless negative samples

### Stage 3

Optional later step:

- joint fine-tuning

This is not the first formal training stage.

Direct one-shot multi-task training is not the official first-version protocol.

## 8. Split Protocol

### 8.1 Main Paper Split

- `1.CSV + 2.CSV` for development
- `3.CSV` for final test

Inside development:

- split by size-depth group
- never by frame

### 8.2 Leakage Rule

Frame-level random splitting is forbidden because heavily overlapping windows would produce leakage.

## 9. Runtime Output Policy

The formal first-version runtime policy is:

- always display `y_det_prob`
- display `size` and `depth` only when `p_det >= threshold`

Displayed outputs:

- `结节概率`
- `最可能大小`
- `大小连续估计值`
- `最可能深度层级`

This gated display is part of the official system contract and should be used by the GUI branch.

## 10. Paper Result Structure

The paper result table should be divided into:

1. `Detection`
2. `Size inversion`
3. `Depth mechanism + coarse depth prediction`

Depth should not be described as a stable high-precision fine-grained inversion result unless a later 6-class protocol clearly succeeds.

## 11. Canonical Code Reference

The canonical code-side protocol definition is:

- `models/task_protocol_v1.py`

All future training scripts, evaluation scripts, and GUI integration code should import the constants and helper functions from that module instead of redefining axis lengths, coarse depth mappings, or runtime gate behavior locally.

The first formal code-level protocol is mirrored in:

- `models/task_protocol_v1.py`
