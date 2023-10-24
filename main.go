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

// Random is a random variable
type Random struct {
	Mean   float64
	Stddev float64
}

// Distribution is a distribution of a neural network
type Distribution struct {
	EncoderWeights []Random
	EncoderBias    []Random
	DecoderWeights []Random
	DecoderBias    []Random
}

// NewDistribution creates a new distribution
func NewDistribution(rng *rand.Rand) Distribution {
	var d Distribution
	factor := math.Sqrt(2.0 / float64(EncoderCols))
	for i := 0; i < EncoderSize; i++ {
		d.EncoderWeights = append(d.EncoderWeights, Random{
			Mean:   factor * rng.NormFloat64(),
			Stddev: factor * rng.NormFloat64(),
		})
	}
	for i := 0; i < EncoderRows; i++ {
		d.EncoderBias = append(d.EncoderBias, Random{
			Mean:   factor * rng.NormFloat64(),
			Stddev: factor * rng.NormFloat64(),
		})
	}
	factor = math.Sqrt(2.0 / float64(DecoderCols))
	for i := 0; i < DecoderSize; i++ {
		d.DecoderWeights = append(d.DecoderWeights, Random{
			Mean:   factor * rng.NormFloat64(),
			Stddev: factor * rng.NormFloat64(),
		})
	}
	for i := 0; i < DecoderRows; i++ {
		d.DecoderBias = append(d.DecoderBias, Random{
			Mean:   factor * rng.NormFloat64(),
			Stddev: factor * rng.NormFloat64(),
		})
	}
	return d
}

// Network is a neural network
type Network struct {
	EncoderState   Matrix
	EncoderWeights Matrix
	EncoderBias    Matrix
	DecoderState   Matrix
	DecoderWeights Matrix
	DecoderBias    Matrix
	Loss           float64
}

// Sample samples a network from the distribution
func (d Distribution) Sample(rng *rand.Rand) Network {
	var n Network
	n.EncoderState = NewMatrix(0, EncoderCols, 1)
	n.EncoderState.Data = n.EncoderState.Data[:EncoderCols]
	n.EncoderWeights = NewMatrix(0, EncoderCols, EncoderRows)
	n.EncoderBias = NewMatrix(0, 1, EncoderRows)
	for i := 0; i < EncoderSize; i++ {
		r := d.EncoderWeights[i]
		n.EncoderWeights.Data = append(n.EncoderWeights.Data, rng.NormFloat64()*r.Stddev+r.Mean)
	}
	for i := 0; i < EncoderRows; i++ {
		r := d.EncoderBias[i]
		n.EncoderBias.Data = append(n.EncoderBias.Data, rng.NormFloat64()*r.Stddev+r.Mean)
	}

	n.DecoderState = NewMatrix(0, DecoderCols, 1)
	n.DecoderState.Data = n.DecoderState.Data[:DecoderCols]
	n.DecoderWeights = NewMatrix(0, DecoderCols, DecoderRows)
	n.DecoderBias = NewMatrix(0, 1, DecoderRows)
	for i := 0; i < DecoderSize; i++ {
		r := d.DecoderWeights[i]
		n.DecoderWeights.Data = append(n.DecoderWeights.Data, rng.NormFloat64()*r.Stddev+r.Mean)
	}
	for i := 0; i < DecoderRows; i++ {
		r := d.DecoderBias[i]
		n.DecoderBias.Data = append(n.DecoderBias.Data, rng.NormFloat64()*r.Stddev+r.Mean)
	}

	return n
}

// Inference run inference on the network
func (n *Network) Inference(data []byte) {
	for _, symbol := range data {
		for i := 0; i < 256; i++ {
			n.EncoderState.Data[Offset+i] = 0
		}
		n.EncoderState.Data[Offset+int(symbol)] = 1
		output := Everett(Add(Mul(n.EncoderWeights, n.EncoderState), n.EncoderBias))
		copy(n.EncoderState.Data[:Offset], output.Data)
	}
	copy(n.DecoderState.Data, n.EncoderState.Data[:Offset])
	loss := 0.0
	for _, symbol := range data {
		output := Everett(Add(Mul(n.DecoderWeights, n.DecoderState), n.DecoderBias))
		copy(n.DecoderState.Data, output.Data[:Offset])
		expected := make([]float64, 512)
		expected[2*int(symbol)] = 1
		for i := 0; i < 512; i++ {
			diff := expected[i] - output.Data[Offset+i]
			loss += diff * diff
		}
	}
	n.Loss = loss
}

func main() {
	rng := rand.New(rand.NewSource(1))
	data, err := ioutil.ReadFile("pg10.txt")
	if err != nil {
		panic(err)
	}

	data = data[:1024]

	distribution := NewDistribution(rng)
	sample := distribution.Sample(rng)
	sample.Inference(data)
}
