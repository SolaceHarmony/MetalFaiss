# Swift Faiss

[![](https://img.shields.io/endpoint?url=https%3A%2F%2Fswiftpackageindex.com%2Fapi%2Fpackages%2Fjkrukowski%2FSwiftFaiss%2Fbadge%3Ftype%3Dswift-versions)](https://swiftpackageindex.com/jkrukowski/SwiftFaiss)
[![](https://img.shields.io/endpoint?url=https%3A%2F%2Fswiftpackageindex.com%2Fapi%2Fpackages%2Fjkrukowski%2FSwiftFaiss%2Fbadge%3Ftype%3Dplatforms)](https://swiftpackageindex.com/jkrukowski/SwiftFaiss)

Use [Faiss](https://github.com/facebookresearch/faiss) in Swift.

Based on [Faiss Mobile](https://github.com/DeveloperMindset-com/faiss-mobile) and [OpenMP Mobile](https://github.com/DeveloperMindset-com/openmp-mobile).

## Usage

Command line demo

```
$ swift run swift-faiss <subcommand> <options>
```

Available subcommands:

- `flat`: create a `FlatIndex`, add vectors to it and search for the most similar sentences.
- `ivfflat`: create an `IVFFlatIndex`, train and add vectors to it and search for the most similar sentences.
- `pq`: create an `PQIndex`, train and add vectors to it and search for the most similar sentences.
- `clustering`: k-means clustering example.

Command line help

```
$ swift run swift-faiss --help
```

In your own code

```swift
import SwiftFaiss

let embeddings: [[Float]] = [
    [0.1, 0.2, 0.3],
    [0.4, 0.5, 0.6],
    [0.7, 0.8, 0.9],
    [1.0, 1.1, 1.2],
    [1.3, 1.4, 1.5],
    [1.6, 1.7, 1.8]
]
let d = embeddings[0].count
let index = try FlatIndex(d: d, metricType: .l2)
try index.add(embeddings)

let result = try index.search([[0.1, 0.5, 0.9]], k: 2)
// do something with result
```

https://github.com/jkrukowski/UseSwiftFaiss contains an iOS example.

## Installation

### Swift Package Manager

You can use [Swift Package Manager](https://swift.org/package-manager/) and specify dependency in `Package.swift` by adding:

```swift
.package(url: "https://github.com/jkrukowski/SwiftFaiss.git", from: "0.0.7")
```

## Format code

```
$ swift package plugin --allow-writing-to-package-directory swiftformat
```

## Tests

```
$ swift test
```

## More info

- [Faiss: The Missing Manual](https://www.pinecone.io/learn/series/faiss/)
- [Faiss C API](https://github.com/facebookresearch/faiss/blob/main/c_api/INSTALL.md)

## Python Implementation

### Usage

Command line demo

```
$ python3 <script_name>.py <options>
```

Available scripts:

- `Clustering.py`: create a k-means clustering model, train it, and predict clusters.
- `FlatIndex.py`: create a `FlatIndex`, add vectors to it, and search for the most similar vectors.
- `IVFFlatIndex.py`: create an `IVFFlatIndex`, train and add vectors to it, and search for the most similar vectors.
- `PQIndex.py`: create a `PQIndex`, train and add vectors to it, and search for the most similar vectors.
- `PreTransformIndex.py`: create a `PreTransformIndex`, train and add vectors to it, and search for the most similar vectors.

### Installation

1. Clone the repository:

```
$ git clone https://github.com/sydneyrenee/MetalFaiss.git
$ cd MetalFaiss
```

2. Install the required dependencies:

```
$ pip install -r requirements.txt
```

3. Run the desired script:

```
$ python3 <script_name>.py <options>
```

### Note

The Python implementation now uses MLX routines instead of NumPy for array and tensor operations.

### Lazy Evaluation

#### Why Lazy Evaluation

When you perform operations in MLX, no computation actually happens. Instead a compute graph is recorded. The actual computation only happens if an eval() is performed.

MLX uses lazy evaluation because it has some nice features, some of which we describe below.

#### Transforming Compute Graphs

Lazy evaluation lets us record a compute graph without actually doing any computations. This is useful for function transformations like grad() and vmap() and graph optimizations.

Currently, MLX does not compile and rerun compute graphs. They are all generated dynamically. However, lazy evaluation makes it much easier to integrate compilation for future performance enhancements.

#### Only Compute What You Use

In MLX you do not need to worry as much about computing outputs that are never used. For example:

```python
def fun(x):
    a = fun1(x)
    b = expensive_fun(a)
    return a, b

y, _ = fun(x)
```

Here, we never actually compute the output of expensive_fun. Use this pattern with care though, as the graph of expensive_fun is still built, and that has some cost associated to it.

Similarly, lazy evaluation can be beneficial for saving memory while keeping code simple. Say you have a very large model Model derived from mlx.nn.Module. You can instantiate this model with model = Model(). Typically, this will initialize all of the weights as float32, but the initialization does not actually compute anything until you perform an eval(). If you update the model with float16 weights, your maximum consumed memory will be half that required if eager computation was used instead.

This pattern is simple to do in MLX thanks to lazy computation:

```python
model = Model() # no memory used yet
model.load_weights("weights_fp16.safetensors")
```

#### When to Evaluate

A common question is when to use eval(). The trade-off is between letting graphs get too large and not batching enough useful work.

For example:

```python
for _ in range(100):
     a = a + b
     mx.eval(a)
     b = b * 2
     mx.eval(b)
```

This is a bad idea because there is some fixed overhead with each graph evaluation. On the other hand, there is some slight overhead which grows with the compute graph size, so extremely large graphs (while computationally correct) can be costly.

Luckily, a wide range of compute graph sizes work pretty well with MLX: anything from a few tens of operations to many thousands of operations per evaluation should be okay.

Most numerical computations have an iterative outer loop (e.g. the iteration in stochastic gradient descent). A natural and usually efficient place to use eval() is at each iteration of this outer loop.

Here is a concrete example:

```python
for batch in dataset:

    # Nothing has been evaluated yet
    loss, grad = value_and_grad_fn(model, batch)

    # Still nothing has been evaluated
    optimizer.update(model, grad)

    # Evaluate the loss and the new parameters which will
    # run the full gradient computation and optimizer update
    mx.eval(loss, model.parameters())
```

An important behavior to be aware of is when the graph will be implicitly evaluated. Anytime you print an array, convert it to an numpy.ndarray, or otherwise access its memory via memoryview, the graph will be evaluated. Saving arrays via save() (or any other MLX saving functions) will also evaluate the array.

Calling array.item() on a scalar array will also evaluate it. In the example above, printing the loss (print(loss)) or adding the loss scalar to a list (losses.append(loss.item())) would cause a graph evaluation. If these lines are before mx.eval(loss, model.parameters()) then this will be a partial evaluation, computing only the forward pass.

Also, calling eval() on an array or set of arrays multiple times is perfectly fine. This is effectively a no-op.
