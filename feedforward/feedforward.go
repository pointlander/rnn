// Copyright 2023 The RNN Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package feedforward

import (
	"fmt"
	"math"
	"math/rand"
	"runtime"
	"sort"

	"github.com/pointlander/datum/iris"
	"github.com/pointlander/rnn/recurrent"
)

const (
	// Window is the distribution window
	Window = 8
	// Middle is the width of the middle layer
	Middle = 4
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

// NewDistrution creates a new distribution of feed forward layers
func NewDistribution(rng *rand.Rand) Distribution {
	layer1Weights := make([]Random, 0, 4*Middle)
	//factor := math.Sqrt(2.0 / float64(4))
	for i := 0; i < 4*Middle; i++ {
		layer1Weights = append(layer1Weights, Random{
			Mean:   0, //factor * rng.NormFloat64(),
			Stddev: 1, //factor * rng.NormFloat64(),
		})
	}
	layer1Bias := make([]Random, 0, Middle)
	for i := 0; i < Middle; i++ {
		layer1Bias = append(layer1Bias, Random{
			Mean:   0, //factor * rng.NormFloat64(),
			Stddev: 1, //factor * rng.NormFloat64(),
		})
	}
	//factor = math.Sqrt(2.0 / float64(Middle))
	layer2Weights := make([]Random, 0, Middle*3)
	for i := 0; i < Middle*3; i++ {
		layer2Weights = append(layer2Weights, Random{
			Mean:   0, //factor * rng.NormFloat64(),
			Stddev: 1, //factor * rng.NormFloat64(),
		})
	}
	//factor = math.Sqrt(2.0 / float64(3))
	layer2Bias := make([]Random, 0, 3)
	for i := 0; i < 3; i++ {
		layer2Bias = append(layer2Bias, Random{
			Mean:   0, //factor * rng.NormFloat64(),
			Stddev: 1, //factor * rng.NormFloat64(),
		})
	}
	return Distribution{
		Layer1Weights: layer1Weights,
		Layer1Bias:    layer1Bias,
		Layer2Weights: layer2Weights,
		Layer2Bias:    layer2Bias,
	}
}

// Sample returns a sampled feedforward neural network
func (d Distribution) Sample(rng *rand.Rand) Sample {
	var s Sample
	s.Layer1Weights = recurrent.NewMatrix32(0, 4, Middle)
	s.Layer1Bias = recurrent.NewMatrix32(0, 1, Middle)
	for i := 0; i < 4*Middle; i++ {
		r := d.Layer1Weights[i]
		s.Layer1Weights.Data = append(s.Layer1Weights.Data, float32(rng.NormFloat64()*r.Stddev+r.Mean))
	}
	for i := 0; i < Middle; i++ {
		r := d.Layer1Bias[i]
		s.Layer1Bias.Data = append(s.Layer1Bias.Data, float32(rng.NormFloat64()*r.Stddev+r.Mean))
	}

	s.Layer2Weights = recurrent.NewMatrix32(0, Middle, 3)
	s.Layer2Bias = recurrent.NewMatrix32(0, 1, 3)
	for i := 0; i < Middle*3; i++ {
		r := d.Layer2Weights[i]
		s.Layer2Weights.Data = append(s.Layer2Weights.Data, float32(rng.NormFloat64()*r.Stddev+r.Mean))
	}
	for i := 0; i < 3; i++ {
		r := d.Layer2Bias[i]
		s.Layer2Bias.Data = append(s.Layer2Bias.Data, float32(rng.NormFloat64()*r.Stddev+r.Mean))
	}

	return s
}

// Learn learn the mode
func Learn() {
	rng := rand.New(rand.NewSource(1))
	data, err := iris.Load()
	if err != nil {
		panic(err)
	}

	for _, value := range data.Fisher {
		sum := 0.0
		for _, v := range value.Measures {
			sum += v * v
		}
		length := math.Sqrt(sum)
		for i := range value.Measures {
			value.Measures[i] /= length
		}
	}

	distribution := NewDistribution(rng)
	networks := make([]Sample, 150)
	minLoss := math.MaxFloat64
	done := make(chan bool, 8)
	cpus := runtime.NumCPU()
	best := Sample{}
	inference := func(seed int64, j int) {
		//rng := rand.New(rand.NewSource(seed))
		loss := 0.0
		for i := 0; i < 150; i++ {
			fisher := data.Fisher[i]
			input := recurrent.NewMatrix32(0, 4, 1)
			for _, v := range fisher.Measures {
				input.Data = append(input.Data, float32(v))
			}
			output := recurrent.Sigmoid32(recurrent.Add32(recurrent.Mul32(networks[j].Layer1Weights, input),
				networks[j].Layer1Bias))
			output = recurrent.Add32(recurrent.Mul32(networks[j].Layer2Weights, output), networks[j].Layer2Bias)
			expected := make([]float32, 3)
			expected[iris.Labels[fisher.Label]] = 1

			for i, v := range output.Data {
				diff := float64(v - expected[i])
				loss += diff * diff
			}
		}
		networks[j].Loss = loss
		done <- true
	}
	for i := 0; i < 2*1024; i++ {
		for j := range networks {
			networks[j] = distribution.Sample(rng)
		}
		k, flight := 0, 0
		for j := 0; j < cpus && k < len(networks); j++ {
			go inference(rng.Int63(), k)
			flight++
			k++
		}
		for k < len(networks) {
			<-done
			flight--
			go inference(rng.Int63(), k)
			flight++
			k++
		}
		for flight > 0 {
			<-done
			flight--
		}
		sort.Slice(networks, func(i, j int) bool {
			return networks[i].Loss < networks[j].Loss
		})
		min, index := math.MaxFloat64, 0
		for j := 0; j < 150-Window; j++ {
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
			Layer1Weights: make([]Random, len(distribution.Layer1Weights)),
			Layer1Bias:    make([]Random, len(distribution.Layer1Bias)),
			Layer2Weights: make([]Random, len(distribution.Layer2Weights)),
			Layer2Bias:    make([]Random, len(distribution.Layer2Bias)),
		}
		for j := 0; j < Window; j++ {
			for k, value := range networks[index+j].Layer1Weights.Data {
				next.Layer1Weights[k].Mean += float64(value)
			}
			for k, value := range networks[index+j].Layer1Bias.Data {
				next.Layer1Bias[k].Mean += float64(value)
			}
			for k, value := range networks[index+j].Layer2Weights.Data {
				next.Layer2Weights[k].Mean += float64(value)
			}
			for k, value := range networks[index+j].Layer2Bias.Data {
				next.Layer2Bias[k].Mean += float64(value)
			}
		}
		for j := range next.Layer1Weights {
			next.Layer1Weights[j].Mean /= Window
		}
		for j := range next.Layer1Bias {
			next.Layer1Bias[j].Mean /= Window
		}
		for j := range next.Layer2Weights {
			next.Layer2Weights[j].Mean /= Window
		}
		for j := range next.Layer2Bias {
			next.Layer2Bias[j].Mean /= Window
		}
		for j := 0; j < Window; j++ {
			for k, value := range networks[index+j].Layer1Weights.Data {
				diff := next.Layer1Weights[k].Mean - float64(value)
				next.Layer1Weights[k].Stddev += diff * diff
			}
			for k, value := range networks[index+j].Layer1Bias.Data {
				diff := next.Layer1Bias[k].Mean - float64(value)
				next.Layer1Bias[k].Stddev += diff * diff
			}
			for k, value := range networks[index+j].Layer2Weights.Data {
				diff := next.Layer2Weights[k].Mean - float64(value)
				next.Layer2Weights[k].Stddev += diff * diff
			}
			for k, value := range networks[index+j].Layer2Bias.Data {
				diff := next.Layer2Bias[k].Mean - float64(value)
				next.Layer2Bias[k].Stddev += diff * diff
			}
		}
		for j := range next.Layer1Weights {
			next.Layer1Weights[j].Stddev /= Window
			next.Layer1Weights[j].Stddev = math.Sqrt(next.Layer1Weights[j].Stddev)
		}
		for j := range next.Layer1Bias {
			next.Layer1Bias[j].Stddev /= Window
			next.Layer1Bias[j].Stddev = math.Sqrt(next.Layer1Bias[j].Stddev)
		}
		for j := range next.Layer2Weights {
			next.Layer2Weights[j].Stddev /= Window
			next.Layer2Weights[j].Stddev = math.Sqrt(next.Layer2Weights[j].Stddev)
		}
		for j := range next.Layer2Bias {
			next.Layer2Bias[j].Stddev /= Window
			next.Layer2Bias[j].Stddev = math.Sqrt(next.Layer2Bias[j].Stddev)
		}
		distribution = next
	}

	correct := 0
	loss := 0.0
	for _, fisher := range data.Fisher {
		input := recurrent.NewMatrix32(0, 4, 1)
		for _, v := range fisher.Measures {
			input.Data = append(input.Data, float32(v))
		}

		output := recurrent.Sigmoid32(recurrent.Add32(recurrent.Mul32(best.Layer1Weights, input),
			best.Layer1Bias))
		output = recurrent.Add32(recurrent.Mul32(best.Layer2Weights, output), best.Layer2Bias)
		max, index := float32(0.0), 0
		for i, value := range output.Data {
			if value > max {
				max, index = value, i
			}
		}
		fmt.Println(index, max)
		if index == iris.Labels[fisher.Label] {
			correct++
		}

		expected := make([]float32, 3)
		expected[iris.Labels[fisher.Label]] = 1

		for i, v := range output.Data {
			diff := float64(v - expected[i])
			loss += diff * diff
		}
	}
	fmt.Println("correct", correct, float64(correct)/150)
	fmt.Println("loss", loss)
}