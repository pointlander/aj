// Copyright 2021 The AJ Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"flag"
	"fmt"
	"math"
	"math/rand"
	"sort"
	"time"

	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"
	"gonum.org/v1/plot/vg/draw"

	"github.com/pointlander/datum/iris"
	"github.com/pointlander/gradient/tf32"
	"github.com/pointlander/pagerank"
)

// Mode is the mode to operate in
var Mode = flag.String("mode", "iris", "operation mode")

func main() {
	flag.Parse()

	if *Mode == "iris" {
		headers := []string{
			"sepal length in cm",
			"sepal width in cm",
			"petal length in cm",
			"petal width in cm",
		}

		datum, err := iris.Load()
		if err != nil {
			panic(err)
		}
		fmt.Println("flowers", len(datum.Fisher))
		data := tf32.NewV(4, len(datum.Fisher))
		for _, flower := range datum.Fisher {
			for _, value := range flower.Measures {
				data.X = append(data.X, float32(value))
			}
		}

		Process(headers, &data)
	}

}

// Process processes the data
func Process(headers []string, data *tf32.V) {
	rand.Seed(1)

	width := data.S[0]

	set := tf32.NewSet()
	set.Add("aw1", width, width)
	set.Add("ab1", width)

	for i := range set.Weights {
		w := set.Weights[i]
		if w.S[1] == 1 {
			for i := 0; i < cap(w.X); i++ {
				w.X = append(w.X, 0)
			}
		} else {
			factor := float32(math.Sqrt(2 / float64(w.S[0])))
			for i := 0; i < cap(w.X); i++ {
				w.X = append(w.X, float32(rand.NormFloat64())*factor)
			}
		}
	}

	deltas := make([][]float32, 0, 8)
	for _, p := range set.Weights {
		deltas = append(deltas, make([]float32, len(p.X)))
	}

	l1 := tf32.Add(tf32.Mul(set.Get("aw1"), data.Meta()), set.Get("ab1"))
	cost := tf32.Avg(tf32.Quadratic(data.Meta(), l1))

	iterations := 256
	points := make(plotter.XYs, 0, iterations)
	start := time.Now()
	for i := 0; i < iterations; i++ {
		set.Zero()
		data.Zero()

		total := tf32.Gradient(cost).X[0]
		norm := float32(0)
		for _, p := range set.Weights {
			for _, d := range p.D {
				norm += d * d
			}
		}

		norm = float32(math.Sqrt(float64(norm)))
		scaling := float32(1)
		if norm > 1 {
			scaling = 1 / norm
		}

		alpha, eta := float32(.3), float32(.3)
		for k, p := range set.Weights {
			for l, d := range p.D {
				deltas[k][l] = alpha*deltas[k][l] - eta*d*scaling
				p.X[l] += deltas[k][l]
			}
		}

		points = append(points, plotter.XY{X: float64(i), Y: float64(total)})
		fmt.Println(i, total)
	}
	fmt.Println(time.Now().Sub(start))

	p := plot.New()

	p.Title.Text = "epochs vs cost"
	p.X.Label.Text = "epochs"
	p.Y.Label.Text = "cost"

	scatter, err := plotter.NewScatter(points)
	if err != nil {
		panic(err)
	}
	scatter.GlyphStyle.Radius = vg.Length(1)
	scatter.GlyphStyle.Shape = draw.CircleGlyph{}
	p.Add(scatter)

	err = p.Save(8*vg.Inch, 8*vg.Inch, "epochs.png")
	if err != nil {
		panic(err)
	}

	aw1 := set.ByName["aw1"]
	graph := pagerank.NewGraph32()
	for i := 0; i < width; i++ {
		for j := 0; j < width; j++ {
			weight := aw1.X[i*width+j]
			if weight < 0 {
				graph.Link(uint64(j), uint64(i), -weight)
			} else {
				graph.Link(uint64(i), uint64(j), weight)
			}
		}
	}
	type Rank struct {
		Node uint64
		Rank float32
	}
	ranks := []Rank{}
	graph.Rank(0.85, 0.000001, func(node uint64, rank float32) {
		ranks = append(ranks, Rank{
			Node: node,
			Rank: rank,
		})
	})
	sort.Slice(ranks, func(i, j int) bool {
		return ranks[i].Rank < ranks[j].Rank
	})
	sum := float32(0.0)
	for _, rank := range ranks {
		fmt.Println(rank.Node, headers[rank.Node], rank.Rank)
		sum += rank.Rank
	}
	fmt.Println(sum)
}
