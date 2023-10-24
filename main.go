// Copyright 2023 The RNN Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"io/ioutil"
	"math"
	"math/rand"
)

const (
	// Width is the width of the network
	Width = 256
	// Cols is the number of columns
	Cols = 2*Width + 256
	// Rows is the number of rows
	Rows = Width
	// Offset is the input offset
	Offset = 2 * Width
	// Size is the number of parameters
	Size = Cols * Rows
)

func main() {
	rng := rand.New(rand.NewSource(1))
	data, err := ioutil.ReadFile("pg10.txt")
	if err != nil {
		panic(err)
	}

	data = data[:1024]

	input := NewMatrix(0, Cols, 1)
	input.Data = input.Data[:Cols]
	layer := NewMatrix(0, Cols, Rows)
	bias := NewMatrix(0, 1, Rows)
	bias.Data = bias.Data[:Rows]
	factor := math.Sqrt(2.0 / float64(Cols))
	for i := 0; i < Size; i++ {
		layer.Data = append(layer.Data, factor*rng.NormFloat64())
	}
	for _, symbol := range data {
		for i := 0; i < 256; i++ {
			input.Data[Offset+i] = 0
		}
		input.Data[Offset+int(symbol)] = 1
		output := Everett(Add(Mul(layer, input), bias))
		copy(input.Data[:Offset], output.Data)
	}

	input2 := NewMatrix(0, 2*Width, 1)
	input2.Data = input2.Data[:2*Width]
	layer2 := NewMatrix(0, 2*Width, Width+256)
	bias2 := NewMatrix(0, 1, Width+256)
	bias2.Data = bias2.Data[:Width+256]
	factor2 := math.Sqrt(2.0 / float64(2*Width))
	for i := 0; i < 2*Width*(Width+256); i++ {
		layer2.Data = append(layer2.Data, factor2*rng.NormFloat64())
	}
	copy(input2.Data, input.Data[:Offset])
	for _, symbol := range data {
		_ = symbol
		output := Everett(Add(Mul(layer2, input2), bias2))
		copy(input2.Data, output.Data[:Offset])
	}
}
