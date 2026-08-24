// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Darwin's Cat — Oleh Tsymaienko & Alisa Lafoks. Part of OrbitCapture NAM — see LICENSE.

//go:build darwin && cgo

package tray

/*
#cgo LDFLAGS: -framework Cocoa
#include <stdlib.h>
void oncTraySetupShow(const char* host, int port, const char* db, const char* user,
                      const char* pass, const char* schema, int keepAwake,
                      const void* iconBytes, int iconLen);
*/
import "C"

import (
	"sync"
	"unsafe"
)

// The sheet is modal and there is one menu bar, so there is one save target at a
// time. Held here rather than passed through C: a Go pointer may not travel into
// C code and come back.
var (
	setupMu   sync.Mutex
	setupSave func(Setup)
)

// ShowSetup opens the settings sheet on the main thread and calls save with what
// the person typed, once, if they pressed Save. Cancel calls nothing.
func ShowSetup(cur Setup, save func(Setup)) {
	setupMu.Lock()
	setupSave = save
	setupMu.Unlock()

	cs := func(s string) *C.char { return C.CString(s) }
	host, db, user, pass, schema := cs(cur.Host), cs(cur.Database), cs(cur.User), cs(cur.Password), cs(cur.Schema)
	defer func() {
		for _, p := range []*C.char{host, db, user, pass, schema} {
			C.free(unsafe.Pointer(p))
		}
	}()
	awake := 0
	if cur.KeepAwake {
		awake = 1
	}
	C.oncTraySetupShow(host, C.int(cur.Port), db, user, pass, schema, C.int(awake),
		unsafe.Pointer(&icon[0]), C.int(len(icon)))
}

//export oncSetupSaved
func oncSetupSaved(host *C.char, port C.int, db, user, pass, schema *C.char, keepAwake C.int) {
	setupMu.Lock()
	save := setupSave
	setupMu.Unlock()
	if save == nil {
		return
	}
	save(Setup{
		Host:      C.GoString(host),
		Port:      int(port),
		Database:  C.GoString(db),
		User:      C.GoString(user),
		Password:  C.GoString(pass),
		Schema:    C.GoString(schema),
		KeepAwake: keepAwake != 0,
	})
}
