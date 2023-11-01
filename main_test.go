// Copyright 2023 The RNN Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"math"
	"math/rand"
	"testing"

	"github.com/pointlander/rnn/discrete"
	"github.com/pointlander/rnn/quanta"
	"github.com/pointlander/rnn/recurrent"
)

const program = `
[ This program prints "Hello World!" and a newline to the screen, its
length is 106 active command characters. [It is not the shortest.]

This loop is an "initial comment loop", a simple way of adding a comment
to a BF program such that you don't have to worry about any command
characters. Any ".", ",", "+", "-", "<" and ">" characters are simply
ignored, the "[" and "]" characters just have to be balanced. This
loop and the commands it contains are ignored because the current cell
defaults to a value of 0; the 0 value causes this loop to be skipped.
]
++++++++               Set Cell #0 to 8
[
  >++++               Add 4 to Cell #1; this will always set Cell #1 to 4
  [                   as the cell will be cleared by the loop
	  >++             Add 2 to Cell #2
	  >+++            Add 3 to Cell #3
	  >+++            Add 3 to Cell #4
	  >+              Add 1 to Cell #5
	  <<<<-           Decrement the loop counter in Cell #1
  ]                   Loop until Cell #1 is zero; number of iterations is 4
  >+                  Add 1 to Cell #2
  >+                  Add 1 to Cell #3
  >-                  Subtract 1 from Cell #4
  >>+                 Add 1 to Cell #6
  [<]                 Move back to the first zero cell you find; this will
					  be Cell #1 which was cleared by the previous loop
  <-                  Decrement the loop Counter in Cell #0
]                       Loop until Cell #0 is zero; number of iterations is 8

The result of this is:
Cell no :   0   1   2   3   4   5   6
Contents:   0   0  72 104  88  32   8
Pointer :   ^

>>.                     Cell #2 has value 72 which is 'H'
>---.                   Subtract 3 from Cell #3 to get 101 which is 'e'
+++++++..+++.           Likewise for 'llo' from Cell #3
>>.                     Cell #5 is 32 for the space
<-.                     Subtract 1 from Cell #4 for 87 to give a 'W'
<.                      Cell #3 was set to 'o' from the end of 'Hello'
+++.------.--------.    Cell #3 for 'rl' and 'd'
>>+.                    Add 1 to Cell #5 gives us an exclamation point
>++.                    And finally a newline from Cell #6`

func BenchmarkBF(b *testing.B) {
	bf := discrete.Compile(program, 30000)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		bf.Reset()
		bf.Run()
	}
}

func BenchmarkFloat32(b *testing.B) {
	rng := rand.New(rand.NewSource(1))
	factor := math.Sqrt(2.0 / float64(256))
	layer := recurrent.NewMatrix32(0, 256, 256)
	for i := 0; i < 256*256; i++ {
		layer.Data = append(layer.Data, float32(factor*rng.NormFloat64()))
	}
	bias := recurrent.NewMatrix32(0, 1, 256)
	for i := 0; i < 256; i++ {
		bias.Data = append(bias.Data, float32(factor*rng.NormFloat64()))
	}
	input := recurrent.NewMatrix32(0, 256, 1)
	for i := 0; i < 256; i++ {
		input.Data = append(input.Data, float32(factor*rng.NormFloat64()))
	}
	for i := 0; i < b.N; i++ {
		recurrent.Step32(recurrent.Add32(recurrent.Mul32(layer, input), bias))
	}
}

func BenchmarkFloat64(b *testing.B) {
	rng := rand.New(rand.NewSource(1))
	factor := math.Sqrt(2.0 / float64(256))
	layer := recurrent.NewMatrix(0, 256, 256)
	for i := 0; i < 256*256; i++ {
		layer.Data = append(layer.Data, factor*rng.NormFloat64())
	}
	bias := recurrent.NewMatrix(0, 1, 256)
	for i := 0; i < 256; i++ {
		bias.Data = append(bias.Data, factor*rng.NormFloat64())
	}
	input := recurrent.NewMatrix(0, 256, 1)
	for i := 0; i < 256; i++ {
		input.Data = append(input.Data, factor*rng.NormFloat64())
	}
	for i := 0; i < b.N; i++ {
		recurrent.Step(recurrent.Add(recurrent.Mul(layer, input), bias))
	}
}

func BenchmarkUint8(b *testing.B) {
	rng := rand.New(rand.NewSource(1))
	layer := quanta.NewMatrix8(256, 256)
	for i := 0; i < 256*256; i++ {
		v := -1
		if rng.Intn(2) == 1 {
			v = 1
		}
		layer.Data = append(layer.Data, int8(v))
	}
	bias := quanta.NewMatrix8(1, 256)
	for i := 0; i < 256; i++ {
		v := -1
		if rng.Intn(2) == 1 {
			v = 1
		}
		bias.Data = append(bias.Data, int8(v))
	}
	input := quanta.NewMatrix8(256, 1)
	for i := 0; i < 256; i++ {
		v := -1
		if rng.Intn(2) == 1 {
			v = 1
		}
		input.Data = append(input.Data, int8(v))
	}
	for i := 0; i < b.N; i++ {
		quanta.Layer8(layer, input, bias)
	}
}

func BenchmarkUint64(b *testing.B) {
	rng := rand.New(rand.NewSource(1))
	layer := quanta.NewMatrix(256, 256)
	for i := 0; i < 256*256/64; i++ {
		layer.Data = append(layer.Data, rng.Uint64())
	}
	bias := quanta.NewMatrix(1, 256)
	for i := 0; i < 256/64; i++ {
		bias.Data = append(bias.Data, rng.Uint64())
	}
	input := quanta.NewMatrix(256, 1)
	for i := 0; i < 256/64; i++ {
		input.Data = append(input.Data, rng.Uint64())
	}
	for i := 0; i < b.N; i++ {
		quanta.Layer(layer, input, bias)
	}
}
