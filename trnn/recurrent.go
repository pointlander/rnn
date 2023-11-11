// Copyright 2023 The RNN Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package trnn

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
	// Window is the size of the window
	Window = 8
	// Width is the width of the network
	Width = 256
	// EncoderCols is the number of encoder columns
	EncoderCols = Width
	// EncoderRows is the number of encoder rows
	EncoderRows = Width
	// EncoderSize is the number of encoder parameters
	EncoderSize = EncoderCols * EncoderRows
	// DecoderCols is the number of decoder columns
	DecoderCols = Width
	// DecoderRows is the number of decoder rows
	DecoderRows = Width
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
	Q              []Random
	K              []Random
	V              []Random
	DecoderWeights []Random
	DecoderBias    []Random
}

// NewDistribution creates a new distribution
func NewDistribution(rng *rand.Rand) Distribution {
	var d Distribution
	//factor := math.Sqrt(2.0 / float64(EncoderCols))
	for i := 0; i < EncoderSize; i++ {
		d.EncoderWeights = append(d.EncoderWeights, Random{
			Mean:   0,
			Stddev: .01,
		})
	}
	for i := 0; i < EncoderRows; i++ {
		d.EncoderBias = append(d.EncoderBias, Random{
			Mean:   0,
			Stddev: .01,
		})
	}
	//factor = math.Sqrt(2.0 / float64(2*Width*Width))
	for i := 0; i < 2*Width*Width; i++ {
		d.Q = append(d.Q, Random{
			Mean:   0,
			Stddev: .01,
		})
	}
	for i := 0; i < 2*Width*Width; i++ {
		d.K = append(d.K, Random{
			Mean:   0,
			Stddev: .01,
		})
	}
	for i := 0; i < 2*Width*Width; i++ {
		d.V = append(d.V, Random{
			Mean:   0,
			Stddev: .01,
		})
	}
	//factor = math.Sqrt(2.0 / float64(DecoderCols))
	for i := 0; i < DecoderSize; i++ {
		d.DecoderWeights = append(d.DecoderWeights, Random{
			Mean:   0,
			Stddev: .01,
		})
	}
	for i := 0; i < DecoderRows; i++ {
		d.DecoderBias = append(d.DecoderBias, Random{
			Mean:   0,
			Stddev: .01,
		})
	}
	return d
}

// Network is a neural network
type Network struct {
	EncoderWeights Matrix
	EncoderBias    Matrix
	Q              Matrix
	K              Matrix
	V              Matrix
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
	n.Q = NewMatrix(0, 2*Width, Width)
	for i := 0; i < 2*Width*Width; i++ {
		r := d.Q[i]
		n.Q.Data = append(n.Q.Data, float32(rng.NormFloat64()*r.Stddev+r.Mean))
	}
	n.K = NewMatrix(0, 2*Width, Width)
	for i := 0; i < 2*Width*Width; i++ {
		r := d.K[i]
		n.K.Data = append(n.K.Data, float32(rng.NormFloat64()*r.Stddev+r.Mean))
	}
	n.V = NewMatrix(0, 2*Width, Width)
	for i := 0; i < 2*Width*Width; i++ {
		r := d.V[i]
		n.V.Data = append(n.V.Data, float32(rng.NormFloat64()*r.Stddev+r.Mean))
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
		input := NewMatrix(0, 256, 1)
		input.Data = input.Data[:cap(input.Data)]
		qState := NewMatrix(0, Width, 256)
		qState.Data = qState.Data[:cap(qState.Data)]
		vState := NewMatrix(0, Width, 256)
		vState.Data = vState.Data[:cap(vState.Data)]
		expected := make([]float64, 256)
		index := 0
		x := data[begin:end]
		for s, symbol := range x[:len(x)-1] {
			for i := 0; i < 256; i++ {
				input.Data[i] = 0
			}
			input.Data[int(symbol)] = 1
			encoded := EverettActivation(Add(MulT(n.EncoderWeights, input), n.EncoderBias))
			q := MulT(n.Q, encoded)
			k := MulT(n.K, encoded)
			v := MulT(n.V, encoded)
			for i, v := range q.Data {
				qState.Data[index*Width+i] = v
			}
			for i, v := range v.Data {
				vState.Data[index*Width+i] = v
			}
			a := SelfAttention(qState, k, vState)
			decoded := TaylorSoftmax(Add(MulT(n.DecoderWeights, a), n.DecoderBias))
			for i := range expected {
				expected[i] = 0
			}
			expected[int(x[s+1])] = 1
			sum := 0.0
			for i := 0; i < 256; i++ {
				diff := expected[i] - float64(decoded.Data[i])
				sum += diff * diff
			}
			index = (index + 1) % 256
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

	//data = data[:1024]

	distribution := NewDistribution(rng)
	networks := make([]Network, 128)
	best := Network{}
	minLoss := math.MaxFloat64
	done := make(chan bool, 8)
	cpus := runtime.NumCPU()
	inference := func(data []byte, j int) {
		networks[j].Inference(data)
		done <- true
	}
	for i := 0; i < 128; i++ {
		for j := range networks {
			networks[j] = distribution.Sample(rng)
		}
		k, flight := 0, 0
		for j := 0; j < cpus && k < len(networks); j++ {
			go inference(data, k)
			flight++
			k++
		}
		for k < len(networks) {
			<-done
			fmt.Printf(".")
			flight--
			go inference(data, k)
			flight++
			k++
		}
		for flight > 0 {
			fmt.Printf(".")
			<-done
			flight--
		}
		fmt.Println()
		sort.Slice(networks, func(i, j int) bool {
			return networks[i].Loss < networks[j].Loss
		})
		min, index := math.MaxFloat64, 0
		for j := 0; j < 64-Window; j++ {
			mean := 0.0
			for k := 0; k < Window; k++ {
				mean += networks[j+k].Loss
			}
			mean /= Window
			stddev := 0.0
			for k := 0; k < Window; k++ {
				diff := mean - networks[j+k].Loss
				stddev += diff * diff
			}
			stddev /= Window
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
			Q:              make([]Random, len(distribution.Q)),
			K:              make([]Random, len(distribution.K)),
			V:              make([]Random, len(distribution.V)),
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
			for k, value := range networks[index+j].Q.Data {
				next.Q[k].Mean += float64(value)
			}
			for k, value := range networks[index+j].K.Data {
				next.K[k].Mean += float64(value)
			}
			for k, value := range networks[index+j].V.Data {
				next.V[k].Mean += float64(value)
			}
			for k, value := range networks[index+j].DecoderWeights.Data {
				next.DecoderWeights[k].Mean += float64(value)
			}
			for k, value := range networks[index+j].DecoderBias.Data {
				next.DecoderBias[k].Mean += float64(value)
			}
		}
		for j := range next.EncoderWeights {
			next.EncoderWeights[j].Mean /= Window
		}
		for j := range next.EncoderBias {
			next.EncoderBias[j].Mean /= Window
		}
		for j := range next.Q {
			next.Q[j].Mean /= Window
		}
		for j := range next.K {
			next.K[j].Mean /= Window
		}
		for j := range next.V {
			next.V[j].Mean /= Window
		}
		for j := range next.DecoderWeights {
			next.DecoderWeights[j].Mean /= Window
		}
		for j := range next.DecoderBias {
			next.DecoderBias[j].Mean /= Window
		}
		for j := 0; j < Window; j++ {
			for k, value := range networks[index+j].EncoderWeights.Data {
				diff := next.EncoderWeights[k].Mean - float64(value)
				next.EncoderWeights[k].Stddev += diff * diff
			}
			for k, value := range networks[index+j].EncoderBias.Data {
				diff := next.EncoderBias[k].Mean - float64(value)
				next.EncoderBias[k].Stddev += diff * diff
			}
			for k, value := range networks[index+j].Q.Data {
				diff := next.Q[k].Mean - float64(value)
				next.Q[k].Stddev += diff * diff
			}
			for k, value := range networks[index+j].K.Data {
				diff := next.K[k].Mean - float64(value)
				next.K[k].Stddev += diff * diff
			}
			for k, value := range networks[index+j].V.Data {
				diff := next.V[k].Mean - float64(value)
				next.V[k].Stddev += diff * diff
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
			next.EncoderWeights[j].Stddev /= Window
			next.EncoderWeights[j].Stddev = math.Sqrt(next.EncoderWeights[j].Stddev)
		}
		for j := range next.EncoderBias {
			next.EncoderBias[j].Stddev /= Window
			next.EncoderBias[j].Stddev = math.Sqrt(next.EncoderBias[j].Stddev)
		}
		for j := range next.Q {
			next.Q[j].Stddev /= Window
			next.Q[j].Stddev = math.Sqrt(next.Q[j].Stddev)
		}
		for j := range next.K {
			next.K[j].Stddev /= Window
			next.K[j].Stddev = math.Sqrt(next.K[j].Stddev)
		}
		for j := range next.V {
			next.V[j].Stddev /= Window
			next.V[j].Stddev = math.Sqrt(next.V[j].Stddev)
		}
		for j := range next.DecoderWeights {
			next.DecoderWeights[j].Stddev /= Window
			next.DecoderWeights[j].Stddev = math.Sqrt(next.DecoderWeights[j].Stddev)
		}
		for j := range next.DecoderBias {
			next.DecoderBias[j].Stddev /= Window
			next.DecoderBias[j].Stddev = math.Sqrt(next.DecoderBias[j].Stddev)
		}
		distribution = next
	}
	output, err := os.Create("network.gob")
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

// Infer inference mode
func Infer() {
	in, err := os.Open("network.gob")
	if err != nil {
		panic(err)
	}
	defer in.Close()
	decoder := gob.NewDecoder(in)
	n := Network{}
	err = decoder.Decode(&n)
	if err != nil {
		panic(err)
	}

	input := NewMatrix(0, 256, 1)
	input.Data = input.Data[:cap(input.Data)]
	qState := NewMatrix(0, Width, 256)
	qState.Data = qState.Data[:cap(qState.Data)]
	vState := NewMatrix(0, Width, 256)
	vState.Data = vState.Data[:cap(vState.Data)]
	index := 0
	data := []byte{'G', 'o'}
	for _, symbol := range data {
		for i := 0; i < 256; i++ {
			input.Data[i] = 0
		}
		input.Data[int(symbol)] = 1
		encoded := EverettActivation(Add(MulT(n.EncoderWeights, input), n.EncoderBias))
		q := MulT(n.Q, encoded)
		k := MulT(n.K, encoded)
		v := MulT(n.V, encoded)
		for i, v := range q.Data {
			qState.Data[index*Width+i] = v
		}
		for i, v := range v.Data {
			vState.Data[index*Width+i] = v
		}
		a := SelfAttention(qState, k, vState)
		decoded := TaylorSoftmax(Add(MulT(n.DecoderWeights, a), n.DecoderBias))
		max, sym := 0.0, 0
		for i, s := range decoded.Data {
			if float64(s) > max {
				max, sym = float64(s), i
			}
		}
		fmt.Printf("%c", sym)
		index = (index + 1) % 256
	}
	fmt.Printf("\n")
}
