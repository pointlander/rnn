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
	. "github.com/pointlander/rnn/matrix/f32"
)

const (
	// Window is the distribution window
	Window = 16
	// Middle is the width of the middle layer
	Middle = 16
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
	Multi         []Multi
}

// Sample is a neural network sample
type Sample struct {
	Layer1Weights Matrix
	Layer1Bias    Matrix
	Layer2Weights Matrix
	Layer2Bias    Matrix
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
			Mean:   0,  //factor * rng.NormFloat64(),
			Stddev: .1, //factor * rng.NormFloat64(),
		})
	}
	//factor = math.Sqrt(2.0 / float64(Middle))
	layer2Weights := make([]Random, 0, 2*Middle*3)
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
			Mean:   0,  //factor * rng.NormFloat64(),
			Stddev: .1, //factor * rng.NormFloat64(),
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
	if d.Multi != nil {
		s.Layer1Weights = NewMatrix(0, 4, Middle)
		s.Layer1Bias = NewMatrix(0, 1, Middle)
		s.Layer2Weights = NewMatrix(0, Middle, 3)
		s.Layer2Bias = NewMatrix(0, 1, 3)
		for n := 0; n < Middle; n++ {
			sample := d.Multi[n].Sample(rng)
			index := 0
			for i := 0; i < 4; i++ {
				s.Layer1Weights.Data = append(s.Layer1Weights.Data, sample[index])
				index++
			}
			s.Layer1Bias.Data = append(s.Layer1Bias.Data, sample[index])
		}
		for n := Middle; n < Middle+3; n++ {
			sample := d.Multi[n].Sample(rng)
			index := 0
			for i := 0; i < Middle; i++ {
				s.Layer2Weights.Data = append(s.Layer2Weights.Data, sample[index])
				index++
			}
			s.Layer2Bias.Data = append(s.Layer2Bias.Data, sample[index])
		}
		return s
	}
	s.Layer1Weights = NewMatrix(0, 4, Middle)
	s.Layer1Bias = NewMatrix(0, 1, Middle)
	for i := 0; i < 4*Middle; i++ {
		r := d.Layer1Weights[i]
		s.Layer1Weights.Data = append(s.Layer1Weights.Data, float32(rng.NormFloat64()*r.Stddev+r.Mean))
	}
	for i := 0; i < Middle; i++ {
		r := d.Layer1Bias[i]
		s.Layer1Bias.Data = append(s.Layer1Bias.Data, float32(rng.NormFloat64()*r.Stddev+r.Mean))
	}

	s.Layer2Weights = NewMatrix(0, Middle, 3)
	s.Layer2Bias = NewMatrix(0, 1, 3)
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
	networks := make([]Sample, 1024)
	minLoss := math.MaxFloat64
	done := make(chan bool, 8)
	cpus := runtime.NumCPU()
	best := Sample{}
	inference := func(seed int64, i [3]int, j int) {
		//rng := rand.New(rand.NewSource(seed))
		loss := 0.0
		for _, i := range i {
			fisher := data.Fisher[i]
			input := NewMatrix(0, 4, 1)
			for /*j*/ _, v := range fisher.Measures {
				input.Data = append(input.Data, float32(v))
			}
			output := Step(Add(MulT(networks[j].Layer1Weights, input),
				networks[j].Layer1Bias))
			output = TaylorSoftmax(Add(MulT(networks[j].Layer2Weights, output), networks[j].Layer2Bias))
			expected := make([]float32, 3)
			expected[iris.Labels[fisher.Label]] = 1

			for i, v := range output.Data {
				diff := float64(float32(v) - expected[i])
				loss += diff * diff
			}
		}
		networks[j].Loss = loss
		done <- true
	}
	indexes := [3]int{rng.Intn(50), 50 + rng.Intn(50), 100 + rng.Intn(50)}
	for i := 0; i < 4*1024; i++ {
		for j := range networks {
			networks[j] = distribution.Sample(rng)
		}
		k, flight := 0, 0
		for j := 0; j < cpus && k < len(networks); j++ {
			go inference(rng.Int63(), indexes, k)
			flight++
			k++
		}
		for k < len(networks) {
			<-done
			flight--
			go inference(rng.Int63(), indexes, k)
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

		if networks[0].Loss < minLoss {
			best = networks[0]
			minLoss = networks[0].Loss
			indexes = [3]int{rng.Intn(50), 50 + rng.Intn(50), 100 + rng.Intn(50)}
		} else {
			//fmt.Println("continue", networks[0].Loss)
			continue
		}
		multi := make([]Multi, 0, Middle+3)
		for n := 0; n < Middle; n++ {
			vars := make([][]float32, 4+1)
			for i := range vars {
				vars[i] = make([]float32, Window)
			}
			for j := 0; j < Window; j++ {
				k := 0
				for _, value := range networks[j].Layer1Weights.Data[n*4 : (n+1)*4] {
					vars[k][j] = float32(value)
					k++
				}
				for _, value := range networks[j].Layer1Bias.Data[n : n+1] {
					vars[k][j] = float32(value)
					k++
				}
			}
			multi = append(multi, Factor(vars, false))
		}
		for n := 0; n < 3; n++ {
			vars := make([][]float32, Middle+1)
			for i := range vars {
				vars[i] = make([]float32, Window)
			}
			for j := 0; j < Window; j++ {
				k := 0
				for _, value := range networks[j].Layer2Weights.Data[n*4 : (n+1)*4] {
					vars[k][j] = float32(value)
					k++
				}
				for _, value := range networks[j].Layer2Bias.Data[n : n+1] {
					vars[k][j] = float32(value)
					k++
				}
			}
			multi = append(multi, Factor(vars, false))
		}

		fmt.Println(networks[0].Loss)
		next := Distribution{
			Multi: multi,
		}
		distribution = next
	}

	correct := 0
	loss := 0.0
	for _, fisher := range data.Fisher {
		input := NewMatrix(0, 4, 1)
		for _, v := range fisher.Measures {
			input.Data = append(input.Data, float32(v))
		}

		output := Step(Add(MulT(best.Layer1Weights, input),
			best.Layer1Bias))
		output = TaylorSoftmax(Add(MulT(best.Layer2Weights, output), best.Layer2Bias))
		max, index := float32(0.0), 0
		for i, value := range output.Data {
			v := float32(value)
			if v > max {
				max, index = v, i
			}
		}
		fmt.Println(index, max)
		if index == iris.Labels[fisher.Label] {
			correct++
		}

		expected := make([]float32, 3)
		expected[iris.Labels[fisher.Label]] = 1

		for i, v := range output.Data {
			diff := float64(float32(v) - expected[i])
			loss += diff * diff
		}
	}
	fmt.Println("correct", correct, float64(correct)/150)
	fmt.Println("loss", loss)
}
