// Copyright 2023 The RNN Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"math/rand"

	"github.com/pointlander/datum"
	"github.com/pointlander/rnn/recurrent"
)

// Random is a random variable
type Random struct {
	Mean   float64
	Stddev float64
}

// Distribution is a distribution of a neural network
type Distribution struct {
	Layer1Weights []Random
	Layer1Bias    []Random
	Layer2Weights []Random
	Layer2Bias    []Random
}

// Sample is a neural network sample
type Sample struct {
	Layer1Weights recurrent.Matrix32
	Layer1Bias    recurrent.Matrix32
	Layer2Weights recurrent.Matrix32
	Layer2Bias    recurrent.Matrix32
	Loss          float64
}

// Learn learn the mode
func Learn() {
	rng := rand.New(rand.NewSource(1))
	data, err := datum.Load()
	if err != nil {
		panic(err)
	}
	_, _ = rng, data
}
