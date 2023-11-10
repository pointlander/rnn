// Copyright 2023 The RNN Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"flag"

	"github.com/pointlander/rnn/discrete"
	"github.com/pointlander/rnn/feedforward"
	"github.com/pointlander/rnn/recurrent"
	"github.com/pointlander/rnn/trnn"
)

var (
	// FlagRecurrent recurrent mode
	FlagRecurrent = flag.Bool("recurrent", false, "recurrent mode")
	// FlagTRNN is the transformer recurrent neural network mode
	FlagTRNN = flag.Bool("trnn", false, "transformer recurrent neural network mode")
	// FlagDiscrete discrete mode
	FlagDiscrete = flag.Bool("discrete", false, "discrete mode")
	// FlagForward feedforward mode
	FlagForward = flag.Bool("forward", false, "feedforward mode")
	// FlagComplexForward feedforward mode
	FlagComplexForward = flag.Bool("complexforward", false, "complex feedforward mode")
)

func main() {
	flag.Parse()

	if *FlagTRNN {
		trnn.Learn()
		return
	} else if *FlagRecurrent {
		recurrent.Learn()
		return
	} else if *FlagDiscrete {
		discrete.Learn()
		return
	} else if *FlagForward {
		feedforward.Learn()
		return
	} else if *FlagComplexForward {
		feedforward.ComplexLearn()
		return
	}

	feedforward.QuatLearn()
}
