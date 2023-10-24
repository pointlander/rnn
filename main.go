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
	// Offset is the offset
	Offset = 2 * Width
	// EncoderCols is the number of encoder columns
	EncoderCols = 2*Width + 256
	// EncoderRows is the number of encoder rows
	EncoderRows = Width
	// EncoderSize is the number of encoder parameters
	EncoderSize = EncoderCols * EncoderRows
	// DecoderCols is the number of decoder columns
	DecoderCols = 2 * Width
	// DecoderRows is the number of decoder rows
	DecoderRows = Width + 256
	// DecoderSize is the number of decoder parameters
	DecoderSize = DecoderCols * DecoderRows
)

func main() {
	rng := rand.New(rand.NewSource(1))
	data, err := ioutil.ReadFile("pg10.txt")
	if err != nil {
		panic(err)
	}

	data = data[:1024]

	inpute := NewMatrix(0, EncoderCols, 1)
	inpute.Data = inpute.Data[:EncoderCols]
	layere := NewMatrix(0, EncoderCols, EncoderRows)
	biase := NewMatrix(0, 1, EncoderRows)
	biase.Data = biase.Data[:EncoderRows]
	factor := math.Sqrt(2.0 / float64(EncoderCols))
	for i := 0; i < EncoderSize; i++ {
		layere.Data = append(layere.Data, factor*rng.NormFloat64())
	}
	for _, symbol := range data {
		for i := 0; i < 256; i++ {
			inpute.Data[Offset+i] = 0
		}
		inpute.Data[Offset+int(symbol)] = 1
		output := Everett(Add(Mul(layere, inpute), biase))
		copy(inpute.Data[:Offset], output.Data)
	}

	inputd := NewMatrix(0, DecoderCols, 1)
	inputd.Data = inputd.Data[:DecoderCols]
	layerd := NewMatrix(0, DecoderCols, DecoderRows)
	biasd := NewMatrix(0, 1, DecoderRows)
	biasd.Data = biasd.Data[:DecoderRows]
	factord := math.Sqrt(2.0 / float64(DecoderCols))
	for i := 0; i < DecoderSize; i++ {
		layerd.Data = append(layerd.Data, factord*rng.NormFloat64())
	}
	copy(inputd.Data, inpute.Data[:Offset])
	for _, symbol := range data {
		_ = symbol
		output := Everett(Add(Mul(layerd, inputd), biasd))
		copy(inputd.Data, output.Data[:Offset])
	}
}
