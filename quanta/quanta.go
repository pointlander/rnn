// Copyright 2023 The RNN Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package quanta

import (
	"fmt"
	"math/rand"
	"time"
)

// Learn learns a model
func Learn() {
	rng := rand.New(rand.NewSource(1))
	layer := NewMatrix(256, 256)
	for i := 0; i < 256*256/64; i++ {
		layer.Data = append(layer.Data, rng.Uint64())
	}
	bias := NewMatrix(1, 256)
	for i := 0; i < 256/64; i++ {
		bias.Data = append(bias.Data, rng.Uint64())
	}
	input := NewMatrix(256, 1)
	for i := 0; i < 256/64; i++ {
		input.Data = append(input.Data, rng.Uint64())
	}
	output := Layer(layer, input, bias)
	fmt.Println(output.Data)
	fmt.Println(len(output.Data) * 64)
	start := time.Now()
	for i := 0; i < 1e6; i++ {
		Layer(layer, input, bias)
	}
	fmt.Println("bench", time.Since(start))
}
