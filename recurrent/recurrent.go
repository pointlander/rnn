// Copyright 2023 The RNN Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package recurrent

import (
	"compress/gzip"
	"encoding/gob"
	"fmt"
	"io/ioutil"
	"math"
	"math/rand"
	"os"
	"runtime"
	"sort"

	. "github.com/pointlander/rnn/matrix/f32"
)

const (
	// Width is the width of the network
	Width = 256
	// Offset is the offset
	Offset = Width
	// EncoderCols is the number of encoder columns
	EncoderCols = Width + 256
	// EncoderRows is the number of encoder rows
	EncoderRows = Width
	// EncoderSize is the number of encoder parameters
	EncoderSize = EncoderCols * EncoderRows
	// DecoderCols is the number of decoder columns
	DecoderCols = Width
	// DecoderRows is the number of decoder rows
	DecoderRows = 256
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
	EncoderWeights Matrix
	EncoderBias    Matrix
	DecoderWeights Matrix
	DecoderBias    Matrix
	Loss           float64
}

// Sample samples a network from the distribution
func (d Distribution) Sample(rng *rand.Rand) Network {
	var n Network
	n.EncoderWeights = NewMatrix(0, EncoderCols, EncoderRows)
	n.EncoderBias = NewMatrix(0, 1, EncoderRows)
	for i := 0; i < EncoderSize; i++ {
		r := d.EncoderWeights[i]
		n.EncoderWeights.Data = append(n.EncoderWeights.Data, float32(rng.NormFloat64()*r.Stddev+r.Mean))
	}
	for i := 0; i < EncoderRows; i++ {
		r := d.EncoderBias[i]
		n.EncoderBias.Data = append(n.EncoderBias.Data, float32(rng.NormFloat64()*r.Stddev+r.Mean))
	}
	n.DecoderWeights = NewMatrix(0, DecoderCols, DecoderRows)
	n.DecoderBias = NewMatrix(0, 1, DecoderRows)
	for i := 0; i < DecoderSize; i++ {
		r := d.DecoderWeights[i]
		n.DecoderWeights.Data = append(n.DecoderWeights.Data, float32(rng.NormFloat64()*r.Stddev+r.Mean))
	}
	for i := 0; i < DecoderRows; i++ {
		r := d.DecoderBias[i]
		n.DecoderBias.Data = append(n.DecoderBias.Data, float32(rng.NormFloat64()*r.Stddev+r.Mean))
	}

	return n
}

// Inference run inference on the network
func (n *Network) Inference(data []byte) {
	rng := rand.New(rand.NewSource(1))
	loss := 0.0
	for i := 0; i < 1024; i++ {
		begin := rng.Intn(len(data) - 1024)
		end := begin + 1024
		data := data[begin:end]
		state := NewMatrix(0, EncoderCols, 1)
		state.Data = state.Data[:EncoderCols]
		for i, symbol := range data[:len(data)-1] {
			for i := 0; i < 256; i++ {
				state.Data[Offset+i] = -1
			}
			state.Data[Offset+int(symbol)] = 1
			output := Step(Add(MulT(n.EncoderWeights, state), n.EncoderBias))
			copy(state.Data[:Offset], output.Data)
			direct := Add(MulT(n.DecoderWeights, output), n.DecoderBias)
			expected := make([]float64, 256)
			expected[int(data[i+1])] = 1
			sum := 0.0
			for i := 0; i < 256; i++ {
				diff := expected[i] - float64(direct.Data[i])
				sum += diff * diff
			}
			loss += sum / 256
		}
	}
	n.Loss = loss
}

// Learn learns the mode
func Learn() {
	rng := rand.New(rand.NewSource(1))
	input, err := os.Open("pg10.txt.gz")
	if err != nil {
		panic(err)
	}
	defer input.Close()
	reader, err := gzip.NewReader(input)
	if err != nil {
		panic(err)
	}
	defer reader.Close()
	data, err := ioutil.ReadAll(reader)
	if err != nil {
		panic(err)
	}

	distribution := NewDistribution(rng)
	networks := make([]Network, 128)
	best := Network{}
	minLoss := math.MaxFloat64
	done := make(chan bool, 8)
	cpus := runtime.NumCPU()
	inference := func(j int) {
		networks[j].Inference(data)
		done <- true
	}
	for i := 0; i < 32; i++ {
		for j := range networks {
			networks[j] = distribution.Sample(rng)
		}
		k, flight := 0, 0
		for j := 0; j < cpus && k < len(networks); j++ {
			go inference(k)
			flight++
			k++
		}
		for k < len(networks) {
			<-done
			fmt.Printf(".")
			flight--
			go inference(k)
			flight++
			k++
		}
		for flight > 0 {
			<-done
			fmt.Printf(".")
			flight--
		}
		fmt.Printf("\n")
		sort.Slice(networks, func(i, j int) bool {
			return networks[i].Loss < networks[j].Loss
		})
		min, index := math.MaxFloat64, 0
		for j := 0; j < 64-8; j++ {
			mean := 0.0
			for k := 0; k < 8; k++ {
				mean += networks[j+k].Loss
			}
			mean /= 8
			stddev := 0.0
			for k := 0; k < 8; k++ {
				diff := mean - networks[j+k].Loss
				stddev += diff * diff
			}
			stddev /= 8
			stddev = math.Sqrt(stddev)
			if stddev < min {
				min, index = stddev, j
			}
		}
		if networks[index].Loss < minLoss {
			best = networks[index]
			minLoss = networks[index].Loss
		} else {
			continue
		}
		fmt.Println(min, index, networks[index].Loss)
		next := Distribution{
			EncoderWeights: make([]Random, len(distribution.EncoderWeights)),
			EncoderBias:    make([]Random, len(distribution.EncoderBias)),
			DecoderWeights: make([]Random, len(distribution.DecoderWeights)),
			DecoderBias:    make([]Random, len(distribution.DecoderBias)),
		}
		for j := 0; j < 8; j++ {
			for k, value := range networks[index+j].EncoderWeights.Data {
				next.EncoderWeights[k].Mean += float64(value)
			}
			for k, value := range networks[index+j].EncoderBias.Data {
				next.EncoderBias[k].Mean += float64(value)
			}
			for k, value := range networks[index+j].DecoderWeights.Data {
				next.DecoderWeights[k].Mean += float64(value)
			}
			for k, value := range networks[index+j].DecoderBias.Data {
				next.DecoderBias[k].Mean += float64(value)
			}
		}
		for j := range next.EncoderWeights {
			next.EncoderWeights[j].Mean /= 8
		}
		for j := range next.EncoderBias {
			next.EncoderBias[j].Mean /= 8
		}
		for j := range next.DecoderWeights {
			next.DecoderWeights[j].Mean /= 8
		}
		for j := range next.DecoderBias {
			next.DecoderBias[j].Mean /= 8
		}
		for j := 0; j < 8; j++ {
			for k, value := range networks[index+j].EncoderWeights.Data {
				diff := next.EncoderWeights[k].Mean - float64(value)
				next.EncoderWeights[k].Stddev += diff * diff
			}
			for k, value := range networks[index+j].EncoderBias.Data {
				diff := next.EncoderBias[k].Mean - float64(value)
				next.EncoderBias[k].Stddev += diff * diff
			}
			for k, value := range networks[index+j].DecoderWeights.Data {
				diff := next.DecoderWeights[k].Mean - float64(value)
				next.DecoderWeights[k].Stddev += diff * diff
			}
			for k, value := range networks[index+j].DecoderBias.Data {
				diff := next.DecoderBias[k].Mean - float64(value)
				next.DecoderBias[k].Stddev += diff * diff
			}
		}
		for j := range next.EncoderWeights {
			next.EncoderWeights[j].Stddev /= 8
			next.EncoderWeights[j].Stddev = math.Sqrt(next.EncoderWeights[j].Stddev)
		}
		for j := range next.EncoderBias {
			next.EncoderBias[j].Stddev /= 8
			next.EncoderBias[j].Stddev = math.Sqrt(next.EncoderBias[j].Stddev)
		}
		for j := range next.DecoderWeights {
			next.DecoderWeights[j].Stddev /= 8
			next.DecoderWeights[j].Stddev = math.Sqrt(next.DecoderWeights[j].Stddev)
		}
		for j := range next.DecoderBias {
			next.DecoderBias[j].Stddev /= 8
			next.DecoderBias[j].Stddev = math.Sqrt(next.DecoderBias[j].Stddev)
		}
		distribution = next
	}
	output, err := os.Create("recurrent.gob")
	if err != nil {
		panic(err)
	}
	defer output.Close()
	encoder := gob.NewEncoder(output)
	err = encoder.Encode(best)
	if err != nil {
		panic(err)
	}
}
