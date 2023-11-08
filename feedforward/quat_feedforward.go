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
	"gonum.org/v1/gonum/num/quat"
)

const (
	// QuatWindow is the distribution window
	QuatWindow = 16
	// QuatMiddle is the width of the middle layer
	QuatMiddle = 16
	// QuatCount is the number of samples
	QuatCount = 256
)

// QuatRandom is a random variable
type QuatRandom struct {
	Mean   [4]float64
	Stddev [4]float64
}

// QuatDistribution is a distribution of a neural network
type QuatDistribution struct {
	Layer1Weights []QuatRandom
	Layer1Bias    []QuatRandom
	Layer2Weights []QuatRandom
	Layer2Bias    []QuatRandom
}

// QuatSample is a neural network sample
type QuatSample struct {
	Layer1Weights recurrent.QuatMatrix
	Layer1Bias    recurrent.QuatMatrix
	Layer2Weights recurrent.QuatMatrix
	Layer2Bias    recurrent.QuatMatrix
	Loss          float64
}

// NewQuatDistrution creates a new distribution of feed forward layers
func NewQuatDistribution(rng *rand.Rand) QuatDistribution {
	layer1Weights := make([]QuatRandom, 0, 4*QuatMiddle)
	//factor := math.Sqrt(2.0 / float64(4))
	for i := 0; i < 4*QuatMiddle; i++ {
		layer1Weights = append(layer1Weights, QuatRandom{
			Mean:   [4]float64{0, 0, 0, 0},         //factor * rng.NormFloat64(),
			Stddev: [4]float64{.25, .25, .25, .25}, //factor * rng.NormFloat64(),
		})
	}
	layer1Bias := make([]QuatRandom, 0, QuatMiddle)
	for i := 0; i < QuatMiddle; i++ {
		layer1Bias = append(layer1Bias, QuatRandom{
			Mean:   [4]float64{0, 0, 0, 0},     //factor * rng.NormFloat64(),
			Stddev: [4]float64{.1, .1, .1, .1}, //factor * rng.NormFloat64(),
		})
	}
	//factor = math.Sqrt(2.0 / float64(Middle))
	layer2Weights := make([]QuatRandom, 0, 2*QuatMiddle*3)
	for i := 0; i < 2*QuatMiddle*3; i++ {
		layer2Weights = append(layer2Weights, QuatRandom{
			Mean:   [4]float64{0, 0, 0, 0},         //factor * rng.NormFloat64(),
			Stddev: [4]float64{.25, .25, .25, .25}, //factor * rng.NormFloat64(),
		})
	}
	//factor = math.Sqrt(2.0 / float64(3))
	layer2Bias := make([]QuatRandom, 0, 3)
	for i := 0; i < 3; i++ {
		layer2Bias = append(layer2Bias, QuatRandom{
			Mean:   [4]float64{0, 0, 0, 0},     //factor * rng.NormFloat64(),
			Stddev: [4]float64{.1, .1, .1, .1}, //factor * rng.NormFloat64(),
		})
	}
	return QuatDistribution{
		Layer1Weights: layer1Weights,
		Layer1Bias:    layer1Bias,
		Layer2Weights: layer2Weights,
		Layer2Bias:    layer2Bias,
	}
}

// QuatSample returns a sampled feedforward neural network
func (d QuatDistribution) Sample(rng *rand.Rand) QuatSample {
	var s QuatSample
	s.Layer1Weights = recurrent.NewQuatMatrix(0, 4, QuatMiddle)
	s.Layer1Bias = recurrent.NewQuatMatrix(0, 1, QuatMiddle)
	for i := 0; i < 4*QuatMiddle; i++ {
		r := d.Layer1Weights[i]
		s.Layer1Weights.Data = append(s.Layer1Weights.Data, quat.Number{
			Real: rng.NormFloat64()*r.Stddev[0] + r.Mean[0],
			Imag: rng.NormFloat64()*r.Stddev[1] + r.Mean[1],
			Jmag: rng.NormFloat64()*r.Stddev[2] + r.Mean[2],
			Kmag: rng.NormFloat64()*r.Stddev[3] + r.Mean[3],
		})
	}
	for i := 0; i < QuatMiddle; i++ {
		r := d.Layer1Bias[i]
		s.Layer1Bias.Data = append(s.Layer1Bias.Data, quat.Number{
			Real: rng.NormFloat64()*r.Stddev[0] + r.Mean[0],
			Imag: rng.NormFloat64()*r.Stddev[1] + r.Mean[1],
			Jmag: rng.NormFloat64()*r.Stddev[2] + r.Mean[2],
			Kmag: rng.NormFloat64()*r.Stddev[3] + r.Mean[3],
		})
	}

	s.Layer2Weights = recurrent.NewQuatMatrix(0, 2*QuatMiddle, 3)
	s.Layer2Bias = recurrent.NewQuatMatrix(0, 1, 3)
	for i := 0; i < 2*QuatMiddle*3; i++ {
		r := d.Layer2Weights[i]
		s.Layer2Weights.Data = append(s.Layer2Weights.Data, quat.Number{
			Real: rng.NormFloat64()*r.Stddev[0] + r.Mean[0],
			Imag: rng.NormFloat64()*r.Stddev[1] + r.Mean[1],
			Jmag: rng.NormFloat64()*r.Stddev[2] + r.Mean[2],
			Kmag: rng.NormFloat64()*r.Stddev[3] + r.Mean[3],
		})
	}
	for i := 0; i < 3; i++ {
		r := d.Layer2Bias[i]
		s.Layer2Bias.Data = append(s.Layer2Bias.Data, quat.Number{
			Real: rng.NormFloat64()*r.Stddev[0] + r.Mean[0],
			Imag: rng.NormFloat64()*r.Stddev[1] + r.Mean[1],
			Jmag: rng.NormFloat64()*r.Stddev[2] + r.Mean[2],
			Kmag: rng.NormFloat64()*r.Stddev[3] + r.Mean[3],
		})
	}

	return s
}

// QuatLearn learn the mode
func QuatLearn() {
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

	distribution := NewQuatDistribution(rng)
	networks := make([]QuatSample, QuatCount)
	minLoss := math.MaxFloat64
	done := make(chan bool, 8)
	cpus := runtime.NumCPU()
	best := QuatSample{}
	inference := func(seed int64, j int) {
		//rng := rand.New(rand.NewSource(seed))
		loss := 0.0
		for i := 0; i < 150; i++ {
			fisher := data.Fisher[i]
			input := recurrent.NewQuatMatrix(0, 4, 1)
			for _, v := range fisher.Measures {
				input.Data = append(input.Data, quat.Number{
					Real: v,
					Imag: 0,
					Jmag: 0,
					Kmag: 0,
				})
			}
			output := recurrent.QuatEverettActivation(recurrent.QuatAdd(recurrent.QuatMul(networks[j].Layer1Weights, input),
				networks[j].Layer1Bias))
			output = recurrent.QuatAdd(recurrent.QuatMul(networks[j].Layer2Weights, output), networks[j].Layer2Bias)
			max, index := 0.0, 0
			penalty := 0.0
			for i, v := range output.Data {
				a := quat.Abs(v)
				if a > max {
					max, index = a, i
				}
				if iris.Labels[fisher.Label] != i {
					penalty += a
				}
			}
			if index != iris.Labels[fisher.Label] {
				loss += penalty
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
		for j := 0; j < QuatCount-QuatWindow; j++ {
			mean := 0.0
			for k := 0; k < QuatWindow; k++ {
				mean += networks[j+k].Loss
			}
			mean /= QuatWindow
			stddev := 0.0
			for k := 0; k < QuatWindow; k++ {
				diff := mean - networks[j+k].Loss
				stddev += diff * diff
			}
			stddev /= QuatWindow
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
		next := QuatDistribution{
			Layer1Weights: make([]QuatRandom, len(distribution.Layer1Weights)),
			Layer1Bias:    make([]QuatRandom, len(distribution.Layer1Bias)),
			Layer2Weights: make([]QuatRandom, len(distribution.Layer2Weights)),
			Layer2Bias:    make([]QuatRandom, len(distribution.Layer2Bias)),
		}
		for j := 0; j < QuatWindow; j++ {
			for k, value := range networks[index+j].Layer1Weights.Data {
				next.Layer1Weights[k].Mean[0] += value.Real
				next.Layer1Weights[k].Mean[1] += value.Imag
				next.Layer1Weights[k].Mean[2] += value.Jmag
				next.Layer1Weights[k].Mean[3] += value.Kmag
			}
			for k, value := range networks[index+j].Layer1Bias.Data {
				next.Layer1Bias[k].Mean[0] += value.Real
				next.Layer1Bias[k].Mean[1] += value.Imag
				next.Layer1Bias[k].Mean[2] += value.Jmag
				next.Layer1Bias[k].Mean[3] += value.Kmag
			}
			for k, value := range networks[index+j].Layer2Weights.Data {
				next.Layer2Weights[k].Mean[0] += value.Real
				next.Layer2Weights[k].Mean[1] += value.Imag
				next.Layer2Weights[k].Mean[2] += value.Jmag
				next.Layer2Weights[k].Mean[3] += value.Kmag
			}
			for k, value := range networks[index+j].Layer2Bias.Data {
				next.Layer2Bias[k].Mean[0] += value.Real
				next.Layer2Bias[k].Mean[1] += value.Imag
				next.Layer2Bias[k].Mean[2] += value.Jmag
				next.Layer2Bias[k].Mean[3] += value.Kmag
			}
		}
		for j := range next.Layer1Weights {
			for k := range next.Layer1Weights[j].Mean {
				next.Layer1Weights[j].Mean[k] /= QuatWindow
			}
		}
		for j := range next.Layer1Bias {
			for k := range next.Layer1Bias[j].Mean {
				next.Layer1Bias[j].Mean[k] /= QuatWindow
			}
		}
		for j := range next.Layer2Weights {
			for k := range next.Layer2Weights[j].Mean {
				next.Layer2Weights[j].Mean[k] /= QuatWindow
			}
		}
		for j := range next.Layer2Bias {
			for k := range next.Layer2Bias[j].Mean {
				next.Layer2Bias[j].Mean[k] /= QuatWindow
			}
		}
		for j := 0; j < QuatWindow; j++ {
			for k, value := range networks[index+j].Layer1Weights.Data {
				for l := range next.Layer1Weights[k].Mean {
					v := 0.0
					switch l {
					case 0:
						v = value.Real
					case 1:
						v = value.Imag
					case 2:
						v = value.Jmag
					case 3:
						v = value.Kmag
					}
					diff := next.Layer1Weights[k].Mean[l] - v
					next.Layer1Weights[k].Stddev[l] += diff * diff
				}
			}
			for k, value := range networks[index+j].Layer1Bias.Data {
				for l := range next.Layer1Bias[k].Mean {
					v := 0.0
					switch l {
					case 0:
						v = value.Real
					case 1:
						v = value.Imag
					case 2:
						v = value.Jmag
					case 3:
						v = value.Kmag
					}
					diff := next.Layer1Bias[k].Mean[l] - v
					next.Layer1Bias[k].Stddev[l] += diff * diff
				}
			}
			for k, value := range networks[index+j].Layer2Weights.Data {
				for l := range next.Layer2Weights[k].Mean {
					v := 0.0
					switch l {
					case 0:
						v = value.Real
					case 1:
						v = value.Imag
					case 2:
						v = value.Jmag
					case 3:
						v = value.Kmag
					}
					diff := next.Layer2Weights[k].Mean[l] - v
					next.Layer2Weights[k].Stddev[l] += diff * diff
				}
			}
			for k, value := range networks[index+j].Layer2Bias.Data {
				for l := range next.Layer2Bias[k].Mean {
					v := 0.0
					switch l {
					case 0:
						v = value.Real
					case 1:
						v = value.Imag
					case 2:
						v = value.Jmag
					case 3:
						v = value.Kmag
					}
					diff := next.Layer2Bias[k].Mean[l] - v
					next.Layer2Bias[k].Stddev[l] += diff * diff
				}
			}
		}
		for j := range next.Layer1Weights {
			for k := range next.Layer1Weights[j].Stddev {
				next.Layer1Weights[j].Stddev[k] /= QuatWindow
				next.Layer1Weights[j].Stddev[k] = math.Sqrt(next.Layer1Weights[j].Stddev[k])
			}
		}
		for j := range next.Layer1Bias {
			for k := range next.Layer1Bias[j].Stddev {
				next.Layer1Bias[j].Stddev[k] /= QuatWindow
				next.Layer1Bias[j].Stddev[k] = math.Sqrt(next.Layer1Bias[j].Stddev[k])
			}
		}
		for j := range next.Layer2Weights {
			for k := range next.Layer2Weights[j].Stddev {
				next.Layer2Weights[j].Stddev[k] /= QuatWindow
				next.Layer2Weights[j].Stddev[k] = math.Sqrt(next.Layer2Weights[j].Stddev[k])
			}
		}
		for j := range next.Layer2Bias {
			for k := range next.Layer2Bias[j].Stddev {
				next.Layer2Bias[j].Stddev[k] /= QuatWindow
				next.Layer2Bias[j].Stddev[k] = math.Sqrt(next.Layer2Bias[j].Stddev[k])
			}
		}
		distribution = next
	}

	correct := 0
	loss := 0.0
	for _, fisher := range data.Fisher {
		input := recurrent.NewQuatMatrix(0, 4, 1)
		for _, v := range fisher.Measures {
			input.Data = append(input.Data, quat.Number{
				Real: v,
				Imag: 0,
				Jmag: 0,
				Kmag: 0,
			})
		}

		output := recurrent.QuatEverettActivation(recurrent.QuatAdd(recurrent.QuatMul(best.Layer1Weights, input),
			best.Layer1Bias))
		output = recurrent.QuatAdd(recurrent.QuatMul(best.Layer2Weights, output), best.Layer2Bias)
		max, index := float32(0.0), 0
		for i, value := range output.Data {
			v := float32(quat.Abs(value))
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
			diff := float64(float32(quat.Abs(v)) - expected[i])
			loss += diff * diff
		}
	}
	fmt.Println("correct", correct, float64(correct)/150)
	fmt.Println("loss", loss)
}
