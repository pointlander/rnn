// Copyright 2023 The RNN Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"flag"

	"github.com/pointlander/rnn/discrete"
	"github.com/pointlander/rnn/recurrent"
)

var (
	// FlagRecurrent recurrent mode
	FlagRecurrent = flag.Bool("recurrent", false, "recurrent mode")
)

func main() {
	flag.Parse()

	/*const size = 1024
	rng := rand.New(rand.NewSource(1))
	type Random struct {
		Mean   float64
		Stddev float64
	}
	random := make([]Random, 0, size)
	for i := 0; i < size; i++ {
		random = append(random, Random{
			Mean:   rng.NormFloat64(),
			Stddev: rng.NormFloat64(),
		})
	}
	neuron := make([]int8, size)
	for i := range neuron {
		sampled := rng.NormFloat64()*random[i].Stddev + random[i].Mean
		if sampled > 0 {
			neuron[i] = 1
		} else {
			neuron[i] = -1
		}
	}
	count := 0
	for i := 0; i < 128; i++ {
		sum := 0
		for j := range neuron {
			if rng.Intn(2) == 0 {
				sum += int(neuron[j]) * 1
			} else {
				sum += int(neuron[j]) * -1
			}
		}
		fmt.Println(sum)
		if sum > 0 {
			count++
		}
	}
	fmt.Println(random)
	fmt.Println(neuron)
	fmt.Println("count", count)

	quanta.Learn()*/

	/*rng := rand.New(rand.NewSource(1))
	x := recurrent.NewMatrix32(0, 9, 32)
	x = recurrent.Normalize32(x)
	for i := 0; i < 9*32; i++ {
		x.Data = append(x.Data, float32(rng.NormFloat64()))
	}
	y := recurrent.SelfAttention32(x, x, x)
	for i := 0; i < y.Rows; i++ {
		sum := float32(0.0)
		for j := 0; j < y.Cols; j++ {
			fmt.Printf("%f ", y.Data[i*y.Cols+j])
			sum += y.Data[i*y.Cols+j]
		}
		fmt.Printf("%f\n", sum)
	}*/

	if *FlagRecurrent {
		recurrent.Learn()
		return
	}

	discrete.Learn()
}
