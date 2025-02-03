import SwiftFaissC

public final class AnyClustering: BaseClustering {
    public let indexPointer: IndexPointer
    private var parameters: ClusteringParameters
    
    init(indexPointer: IndexPointer, parameters: ClusteringParameters) {
        self.indexPointer = indexPointer
        self.parameters = parameters
    }
    
    public convenience init(d: Int, k: Int, parameters: ClusteringParameters = .init()) throws {
        var indexPtr: OpaquePointer?
        try IndexError.check(
            faiss_Clustering_new(
                &indexPtr,
                Int32(d),
                Int32(k),
                &parameters.faissClusteringParameters
            )
        )
        self.init(indexPointer: IndexPointer(indexPtr!), parameters: parameters)
    }
    
    deinit {
        if isKnownUniquelyReferenced(&indexPointer) {
            faiss_Clustering_free(indexPointer.pointer)
        }
    }
}

/// Simplified interface for k-means clustering.
///
/// - `xs`: training set
/// - `d`: dimension of the data
/// - `k`: number of output centroids
public func kMeansClustering(
    _ xs: [[Float]],
    d: Int,
    k: Int
) throws -> [[Float]] {
    let clustering = try AnyClustering(d: d, k: k)
    let index = try FlatIndex(d: d, metricType: .l2)
    try clustering.train(xs, index: index)
    return clustering.centroids()
}
