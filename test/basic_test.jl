# Simple test without MATLAB dependency to check test framework
using Test

@testset "Basic Test Framework" begin
    @test 1 + 1 == 2
    @test true
    println("✓ Basic test framework working")
end

println("Test framework validation complete!")
