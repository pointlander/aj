// Copyright 2021 The AJ Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"encoding/csv"
	"flag"
	"fmt"
	"math"
	"math/rand"
	"os"
	"sort"
	"strconv"
	"strings"
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
	} else if *Mode == "fab" {
		input, err := os.Open("uci-secom.csv")
		if err != nil {
			panic(err)
		}
		defer input.Close()
		reader := csv.NewReader(input)
		headers, err := reader.Read()
		if err != nil {
			panic(err)
		}
		line, _ := reader.Read()
		width, height := len(line)-1, 1568-1
		data := tf32.NewV(width, height)
		for line != nil {
			for _, value := range line[1:] {
				parsed := 0.0
				if value != "" {
					parsed, err = strconv.ParseFloat(value, 32)
					if err != nil {
						panic(err)
					}
				}
				data.X = append(data.X, float32(parsed))
			}
			line, _ = reader.Read()
		}

		aw1 := Process(headers, &data)

		type Factor struct {
			Column int
			Weight float32
		}
		factors := []Factor{}
		for i := 0; i < width; i++ {
			index := i*width + width - 1
			factors = append(factors, Factor{
				Column: i,
				Weight: aw1.X[index],
			})
		}
		sort.Slice(factors, func(i, j int) bool {
			a, b := factors[i].Weight, factors[j].Weight
			if a < 0 {
				a = -a
			}
			if b < 0 {
				b = -b
			}
			return a < b
		})
		for _, factor := range factors {
			fmt.Println(factor.Column, factor.Weight)
		}
	} else if *Mode == "bee" {
		input, err := os.Open("bee_data.csv")
		if err != nil {
			panic(err)
		}
		defer input.Close()
		reader := csv.NewReader(input)
		_, err = reader.Read()
		if err != nil {
			panic(err)
		}
		width, height := 50, 4
		data := tf32.NewV(width, height)
		line, err := reader.Read()
		if err != nil {
			panic(err)
		}
		headers, index := []string{}, 0
		for line != nil {
			if index < 50 {
				headers = append(headers, line[5])
				index++
			}
			parsed, err := strconv.ParseFloat(strings.ReplaceAll(line[19], ",", ""), 32)
			if err != nil {
				panic(err)
			}
			data.X = append(data.X, float32(parsed))
			line, _ = reader.Read()
		}

		Process(headers, &data)
	} else if *Mode == "quantum" {
		datum, err := iris.Load()
		if err != nil {
			panic(err)
		}
		min, max := make([]float64, 4), make([]float64, 4)
		for i := range min {
			min[i] = math.MaxFloat32
		}
		for _, flower := range datum.Fisher {
			for i, value := range flower.Measures {
				if value > max[i] {
					max[i] = value
				}
				if value < min[i] {
					min[i] = value
				}
			}
		}
		fmt.Println("max", max)
		fmt.Println("min", min)
		width, height := (1<<16)-1, 1
		data := tf32.NewV(width, height)
		data.X = data.X[:cap(data.X)]

		for _, flower := range datum.Fisher {
			var bucket uint16
			for i, value := range flower.Measures {
				value -= min[i]
				value /= max[i] - min[i]
				bucket <<= 4
				bucket |= 1 << int(4*value)
			}
			data.X[bucket]++
		}
		headers := make([]string, 0, width)
		for i, value := range data.X {
			headers = append(headers, fmt.Sprintf("%d", i))
			if value == 0 {
				continue
			}
			fmt.Println(i, value)
		}

		Process(headers, &data)
	}
}

// Histogram generates a histogram of the weights
func Histogram(name, file string, aw1 *tf32.V) {
	values := make(plotter.Values, 0, 1024)
	for _, value := range aw1.X {
		values = append(values, float64(value))
	}

	p := plot.New()
	p.Title.Text = name
	histogram, err := plotter.NewHist(values, 256)
	if err != nil {
		panic(err)
	}
	p.Add(histogram)
	err = p.Save(8*vg.Inch, 8*vg.Inch, file)
	if err != nil {
		panic(err)
	}
}

// Process processes the data
func Process(headers []string, data *tf32.V) *tf32.V {
	rand.Seed(1)

	width := data.S[0]

	set := tf32.NewSet()
	set.Add("aw1", width, width)
	set.Add("ab1", width)
	aw1 := set.ByName["aw1"]

	for i := range set.Weights {
		w := set.Weights[i]
		if w.S[1] == 1 {
			for i := 0; i < cap(w.X); i++ {
				w.X = append(w.X, 0)
			}
		} else {
			factor := float32(math.Sqrt(2 / float64(w.S[0])))
			for i := 0; i < cap(w.X); i++ {
				w.X = append(w.X, float32(rand.Float64()*2-1)*factor)
			}
		}
	}

	Histogram("Before Weight Histogram", "before_weights_histogram.png", aw1)

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

	Histogram("After Weight Histogram", "after_weights_histogram.png", aw1)

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

	return aw1
}
