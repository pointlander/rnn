// Copyright 2023 The RNN Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"flag"
	"math/rand"

	"github.com/pointlander/rnn/discrete"
	"github.com/pointlander/rnn/encdec"
	"github.com/pointlander/rnn/feedforward"
	"github.com/pointlander/rnn/matrix/f32"
	"github.com/pointlander/rnn/recurrent"
	"github.com/pointlander/rnn/trnn"
)

var (
	// FlagRecurrent recurrent mode
	FlagRecurrent = flag.Bool("recurrent", false, "recurrent mode")
	// FlagEncDec is the encoder decoder mode
	FlagEncDec = flag.Bool("encdec", false, "encoder decoder mode")
	// FlagTRNN is the transformer recurrent neural network mode
	FlagTRNN = flag.Bool("trnn", false, "transformer recurrent neural network mode")
	// FlagDiscrete discrete mode
	FlagDiscrete = flag.Bool("discrete", false, "discrete mode")
	// FlagForward feedforward mode
	FlagForward = flag.Bool("forward", false, "feedforward mode")
	// FlagComplexForward feedforward mode
	FlagComplexForward = flag.Bool("complexforward", false, "complex feedforward mode")
	// FlagInfer inference mode
	FlagInfer = flag.Bool("infer", false, "inference mode")
)

func main() {
	flag.Parse()

	if *FlagTRNN {
		if *FlagInfer {
			trnn.Infer()
			return
		}
		trnn.Learn()
		return
	} else if *FlagRecurrent {
		if *FlagInfer {
			recurrent.Infer()
			return
		}
		recurrent.Learn()
		return
	} else if *FlagEncDec {
		encdec.Learn()
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

	//feedforward.QuatLearn()
	rng := rand.New(rand.NewSource(1))
	vars := make([][]float32, 16)
	for i := range vars {
		vars[i] = make([]float32, 8)
		for j := range vars[i] {
			vars[i][j] = float32(rng.NormFloat64())
		}
	}
	f32.Factor(vars, true)
}
