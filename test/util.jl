using Test: @test, @testset, @inferred
using ZebrafishHMM2023: chunks, split_into_repeated_subsequences

@testset "chunks" begin
    @test chunks(1:54, 5) == [1:11, 12:22, 23:33, 34:44, 45:54]
    @test chunks(2:55, 5) == [2:12, 13:23, 24:34, 35:45, 46:55]
end

@testset "split_into_repeated_subsequences" begin
    @test split_into_repeated_subsequences([1,1,2,2]) == [[1,1],[2,2]]
    seq = reduce(vcat, [fill(rand(1:3), rand(1:20)) for _ = 1:20])
    subseqs = split_into_repeated_subsequences(seq)
    @test reduce(vcat, subseqs) == seq
    @test all(map(length ∘ unique, subseqs) .== 1)
    for (s1, s2) = zip(subseqs[1:end - 1], subseqs[2:end])
        @test first(s1) ≠ first(s2)
    end
end
