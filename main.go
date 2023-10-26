// Copyright 2023 The RNN Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"flag"

	"github.com/pointlander/rnn/recurrent"
)

var (
	// FlagRecurrent recurrent mode
	FlagRecurrent = flag.Bool("recurrent", false, "recurrent mode")
)

func main() {
	flag.Parse()

	if *FlagRecurrent {
		recurrent.Learn()
		return
	}
}
