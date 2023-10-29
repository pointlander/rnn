// Copyright 2023 The RNN Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"math"
	"math/rand"
	"testing"

	"github.com/pointlander/rnn/quanta"
	"github.com/pointlander/rnn/recurrent"
)

func BenchmarkFloat64(b *testing.B) {
	rng := rand.New(rand.NewSource(1))
	factor := math.Sqrt(2.0 / float64(256))
	layer := recurrent.NewMatrix(0, 256, 256)
	for i := 0; i < 256*256; i++ {
		layer.Data = append(layer.Data, factor*rng.NormFloat64())
	}
	bias := recurrent.NewMatrix(0, 1, 256)
	for i := 0; i < 256; i++ {
		bias.Data = append(bias.Data, factor*rng.NormFloat64())
	}
	input := recurrent.NewMatrix(0, 256, 1)
	for i := 0; i < 256; i++ {
		input.Data = append(input.Data, factor*rng.NormFloat64())
	}
	for i := 0; i < b.N; i++ {
		recurrent.Step(recurrent.Add(recurrent.Mul(layer, input), bias))
	}
}

func BenchmarkUint64(b *testing.B) {
	rng := rand.New(rand.NewSource(1))
	layer := quanta.NewMatrix(256, 256)
	for i := 0; i < 256*256/64; i++ {
		layer.Data = append(layer.Data, rng.Uint64())
	}
	bias := quanta.NewMatrix(1, 256)
	for i := 0; i < 256/64; i++ {
		bias.Data = append(bias.Data, rng.Uint64())
	}
	input := quanta.NewMatrix(256, 1)
	for i := 0; i < 256/64; i++ {
		input.Data = append(input.Data, rng.Uint64())
	}
	for i := 0; i < b.N; i++ {
		quanta.Layer(layer, input, bias)
	}
}
