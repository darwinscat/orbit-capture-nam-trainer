// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Darwin's Cat — Oleh Tsymaienko & Alisa Lafoks. Part of OrbitCapture NAM — see LICENSE.

package worker

// THE CONTROL POLL IS GONE. It read this job's row every two seconds looking for a cancel, a stop or
// a request to hear the run as it stands, and it was a second channel answering questions the
// checkpoint write already had to ask. Two channels are two things to keep in agreement.
//
// What each of the three became:
//   cancel — comes back with the checkpoint write (store.Verdict), so it lands at the end of the
//            epoch in progress instead of within two seconds. On a workshop whose epochs are ten to
//            thirty seconds that is the price of one mechanism instead of two.
//   stop   — the same, and it no longer has to arm anything: the weights are already in the library,
//            so stopping is closing the job with what is there.
//   live   — nothing at all. The take's row IS the answer, refreshed every epoch, and the player
//            reads it. Nobody asks a trainer to make a snapshot any more.
