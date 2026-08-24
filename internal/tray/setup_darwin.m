// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Darwin's Cat — Oleh Tsymaienko & Alisa Lafoks. Part of OrbitCapture NAM — see LICENSE.
//
// The Setup sheet behind the menu bar's "Setup…". Six boxes and a checkbox: where the shared library
// is, and whether this machine may sleep while the queue has work. It writes nothing itself — on OK it
// hands the values back to Go, which is where the config file and the restart live.
//
// It must run on the main thread: AppKit says so, and the menu click arrives on a goroutine. Hence the
// dispatch to the main queue, and hence nothing here blocks the caller.

#import <Cocoa/Cocoa.h>
#include "_cgo_export.h"

static NSTextField *labelFor(NSString *text, CGFloat y) {
    NSTextField *l = [[NSTextField alloc] initWithFrame:NSMakeRect(0, y, 86, 20)];
    l.stringValue = text;
    l.bezeled = NO;
    l.drawsBackground = NO;
    l.editable = NO;
    l.selectable = NO;
    l.alignment = NSTextAlignmentRight;
    return l;
}

static NSTextField *fieldFor(NSString *value, CGFloat y, CGFloat width, BOOL secure) {
    NSTextField *f = secure ? [[NSSecureTextField alloc] initWithFrame:NSMakeRect(94, y, width, 22)]
                            : [[NSTextField alloc] initWithFrame:NSMakeRect(94, y, width, 22)];
    f.stringValue = value ?: @"";
    return f;
}

void oncTraySetupShow(const char *host, int port, const char *db, const char *user,
                      const char *pass, const char *schema, int keepAwake) {
    NSString *nsHost   = host   ? @(host)   : @"";
    NSString *nsDb     = db     ? @(db)     : @"";
    NSString *nsUser   = user   ? @(user)   : @"";
    NSString *nsPass   = pass   ? @(pass)   : @"";
    NSString *nsSchema = schema ? @(schema) : @"";
    NSString *nsPort   = [NSString stringWithFormat:@"%d", port];

    dispatch_async(dispatch_get_main_queue(), ^{
        const CGFloat rowH = 28, width = 320;
        NSView *box = [[NSView alloc] initWithFrame:NSMakeRect(0, 0, 94 + width, rowH * 7)];

        // Bottom-up: AppKit's origin is the lower-left corner, and the order below is the order read.
        NSButton *awake = [[NSButton alloc] initWithFrame:NSMakeRect(94, 2, width, 20)];
        [awake setButtonType:NSButtonTypeSwitch];
        awake.title = @"Keep this machine awake while the queue has work";
        awake.state = keepAwake ? NSControlStateValueOn : NSControlStateValueOff;

        NSTextField *fSchema = fieldFor(nsSchema, rowH * 1, width, NO);
        NSTextField *fPass   = fieldFor(nsPass,   rowH * 2, width, YES);
        NSTextField *fUser   = fieldFor(nsUser,   rowH * 3, width, NO);
        NSTextField *fDb     = fieldFor(nsDb,     rowH * 4, width, NO);
        NSTextField *fPort   = fieldFor(nsPort,   rowH * 5, 80,    NO);
        NSTextField *fHost   = fieldFor(nsHost,   rowH * 6, width, NO);

        fSchema.placeholderString = @"public";
        fPass.placeholderString   = @"(none — trust auth)";

        [box addSubview:labelFor(@"host", rowH * 6)];     [box addSubview:fHost];
        [box addSubview:labelFor(@"port", rowH * 5)];     [box addSubview:fPort];
        [box addSubview:labelFor(@"database", rowH * 4)]; [box addSubview:fDb];
        [box addSubview:labelFor(@"user", rowH * 3)];     [box addSubview:fUser];
        [box addSubview:labelFor(@"password", rowH * 2)]; [box addSubview:fPass];
        [box addSubview:labelFor(@"schema", rowH * 1)];   [box addSubview:fSchema];
        [box addSubview:awake];

        NSAlert *a = [[NSAlert alloc] init];
        a.messageText = @"Where the library is";
        a.informativeText = @"Saving restarts the trainer. A job it is running goes back in the queue and "
                             "continues from its last finished epoch, wherever it is picked up.";
        [a addButtonWithTitle:@"Save & restart"];
        [a addButtonWithTitle:@"Cancel"];
        a.accessoryView = box;

        // A background daemon owns no window and is not frontmost; without this the sheet opens behind
        // whatever the person is looking at, and the menu they just used seems to have done nothing.
        [NSApp activateIgnoringOtherApps:YES];
        [box.window makeFirstResponder:fHost];

        if ([a runModal] == NSAlertFirstButtonReturn) {
            oncSetupSaved((char *)fHost.stringValue.UTF8String,
                          fPort.intValue,
                          (char *)fDb.stringValue.UTF8String,
                          (char *)fUser.stringValue.UTF8String,
                          (char *)fPass.stringValue.UTF8String,
                          (char *)fSchema.stringValue.UTF8String,
                          awake.state == NSControlStateValueOn ? 1 : 0);
        }
    });
}
