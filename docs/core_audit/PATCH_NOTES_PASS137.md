# Patch notes — Pass 137: ARM simulation unsigned phase parity

## Problem

The embedded simulation path is the host-side feedback loop for ARM32 behavior, but it used the same signed phase overflow patterns as the hardware path. That makes the simulator a poor oracle for phase-wrap correctness.

## Change

Store simulation active phase as `uq32_t`, centralize phase initialization and modulo advances, and saturate simulated chirp frequency updates.

## Why minimal

The existing simulation structure and performance accounting stay in place. Only the arithmetic domain for phase and chirp advancement is tightened.
