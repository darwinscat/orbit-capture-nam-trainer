// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Darwin's Cat — Oleh Tsymaienko & Alisa Lafoks. Part of OrbitCapture NAM — see LICENSE.

// Package jobs holds the daemon's domain types: job kinds, states, and the small
// pure helpers around them. It depends on nothing else in the tree so both the
// store and the HTTP layer can share one definition of a Job.
package jobs

// Kinds (the design notes).
const (
	KindTrain     = "train"
	KindProbeSelf = "probe_self"
	KindProbeE10  = "probe_e10"
)

// States (the design notes).
const (
	StateQueued    = "queued"
	StateRunning   = "running"
	StateSucceeded = "succeeded"
	StateFailed    = "failed"
)

// Verdicts for probe_self.
const (
	VerdictPass = "pass"
	VerdictFail = "fail"
)

// Fixed probe epoch counts (the design notes).
const (
	ProbeSelfEpochs = 1
	ProbeE10Epochs  = 10
)

// MaxTrainEpochs is a sanity ceiling on a train request.
const MaxTrainEpochs = 10000

// ValidKind reports whether k is one of the three job kinds.
func ValidKind(k string) bool {
	switch k {
	case KindTrain, KindProbeSelf, KindProbeE10:
		return true
	}
	return false
}

// NormalizeEpochs returns the epoch count that actually governs a job of this
// kind: probes are fixed (1 / 10); a train uses the requested value. The result
// is what goes into both the stored row and the content-addressed key, so the
// key never depends on an epochs param the daemon would have ignored.
func NormalizeEpochs(kind string, requested int) int {
	switch kind {
	case KindProbeSelf:
		return ProbeSelfEpochs
	case KindProbeE10:
		return ProbeE10Epochs
	default:
		return requested
	}
}

// IsTerminal reports whether a state is final (no further transitions).
func IsTerminal(state string) bool {
	return state == StateSucceeded || state == StateFailed
}

// StoresModel reports whether a kind produces a downloadable .nam. Probes never do.
func StoresModel(kind string) bool { return kind == KindTrain }

// Job is a row of the jobs table plus the derived has_model flag. Nullable
// columns are pointers so the JSON layer renders absent values as null.
type Job struct {
	Key        string
	Kind       string
	State      string
	Priority   int
	Epochs     int
	Arch       string
	CreatedAt  int64
	StartedAt  *int64
	FinishedAt *int64
	PID        *int64
	Epoch      *int64
	SPerEpoch  *float64
	Verdict    *string
	ESR        *float64
	ErrorCode  *string
	ErrorMsg   *string
	HasModel   bool
}
