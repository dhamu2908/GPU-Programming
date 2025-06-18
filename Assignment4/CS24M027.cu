//GPU Assignment 4 CS24M027 Dhamodharan



#include <iostream>
#include <fstream>
#include <vector>
#include <queue>
#include <algorithm>
#include <climits>
#include <unordered_map>
#include <cuda_runtime.h>
#include <chrono>
#include <cmath>
#include <string>
#include <set>

using namespace std;



#define MAX_VALUE LLONG_MAX
#define THREAD_COUNT 1024            //Threads count
#define MAX_EXECUTION_TIME_MS 595000 // 9 minutes 55 seconds


//Road connection 
struct RoadConnection {
    long long destination;
    long long distance;
    long long throughput;
};

//Location with Occupancy
struct RefugeLocation {
    long long location;
    long long maxOccupancy;
};


//Populated city details 
struct OriginLocation {
    long long location;
    long long youngPopulation;
    long long seniorPopulation;
};

// GPU memory allocations
long long *gpu_roadStartIndices, *gpu_roadEndpoints, *gpu_roadDistances, *gpu_roadThroughputs;
long long *gpu_distances;
long long *gpu_predecessors;
bool *gpu_processed;
long long *gpu_shelterLocations, *gpu_shelterCapacities;


//Initializing search arrays
__global__ void initializeSearchArrays(long long locationCount, long long *distances, long long *predecessors, bool *processed) {
    long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < locationCount) {
        distances[idx] = MAX_VALUE;
        predecessors[idx] = -1;
        processed[idx] = false;
    }
}

//Processing Edges
__global__ void processEdges(long long locationCount, long long *roadStartIndices, long long *roadEndpoints, 
                           long long *roadDistances, long long *distances, 
                           long long *predecessors, bool *processed, long long currentNode) {
    long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < locationCount && !processed[idx] && distances[idx] == distances[currentNode]) {
        currentNode = idx;
        processed[currentNode] = true;
        long long startEdge = roadStartIndices[currentNode];
        long long endEdge = roadStartIndices[currentNode + 1];
        
        for (long long edgeIdx = startEdge; edgeIdx < endEdge; edgeIdx++) {
            long long neighborNode = roadEndpoints[edgeIdx];
            long long edgeLength = roadDistances[edgeIdx];
            long long newDistance = distances[currentNode] + edgeLength;
            
            if (newDistance < distances[neighborNode]) {
                distances[neighborNode] = newDistance;
                predecessors[neighborNode] = currentNode;
            }
        }
    }
}

//Identifying minimum distance
__global__ void identifyMinimumDistance(long long locationCount, long long *distances, bool *processed, 
                                     long long *blockMinDistances, long long *blockMinNodes) {
    __shared__ long long sharedMinDistances[THREAD_COUNT];
    __shared__ long long sharedMinNodes[THREAD_COUNT];
    
    long long threadId = threadIdx.x;
    long long globalIdx = blockIdx.x * blockDim.x + threadIdx.x;
    sharedMinDistances[threadId] = MAX_VALUE;
    sharedMinNodes[threadId] = -1;
    
    if (globalIdx < locationCount && !processed[globalIdx] && distances[globalIdx] < sharedMinDistances[threadId]) {
        sharedMinDistances[threadId] = distances[globalIdx];
        sharedMinNodes[threadId] = globalIdx;
    }
    
    __syncthreads();
    
    for (long long stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadId < stride) {
            if (sharedMinDistances[threadId + stride] < sharedMinDistances[threadId]) {
                sharedMinDistances[threadId] = sharedMinDistances[threadId + stride];
                sharedMinNodes[threadId] = sharedMinNodes[threadId + stride];
            }
        }
        __syncthreads();
    }
    
    if (threadId == 0) {
        blockMinDistances[blockIdx.x] = sharedMinDistances[0];
        blockMinNodes[blockIdx.x] = sharedMinNodes[0];
    }
}


//Function to compute shelter scores
__global__ void computeShelterScores(long long locationCount, long long shelterCount, long long *distances,
                                   long long *shelterLocations, long long *shelterCapacities, 
                                   float *shelterScores, long long maxSeniorDistance) {
    long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < shelterCount) {
        long long shelterLocation = shelterLocations[idx];
        long long capacity = shelterCapacities[idx];
        long long distance = distances[shelterLocation];
        
        if (distance != MAX_VALUE) {
            bool seniorAccessible = (distance <= maxSeniorDistance);
            float accessibilityFactor = seniorAccessible ? 2.0f : 1.0f;
            float distanceFactor = (distance == 0) ? 1.0f : (10000.0f / distance);
            float capacityFactor = capacity;
            
            shelterScores[idx] = accessibilityFactor * distanceFactor * capacityFactor;
        } else {
            shelterScores[idx] = 0.0f;
        }
    }
}

// GPU-accelerated shortest path algorithm
void parallelShortestPath(long long locationCount, const vector<vector<RoadConnection>>& roadNetwork, 
                        long long startLocation, vector<long long>& distances, 
                        vector<long long>& predecessors, long long maxSeniorDistance = 0,
                        vector<RefugeLocation>* shelters = nullptr,
                        vector<pair<long long, float>>* rankedShelters = nullptr) {
    vector<long long> roadStartIndices(locationCount + 1, 0);
    vector<long long> roadEndpoints;
    vector<long long> roadDistances;
    vector<long long> roadThroughputs;
    
    for (long long location = 0; location < locationCount; location++) {
        roadStartIndices[location + 1] = roadStartIndices[location] + roadNetwork[location].size();
        for (const RoadConnection& road : roadNetwork[location]) {
            roadEndpoints.push_back(road.destination);
            roadDistances.push_back(road.distance);
            roadThroughputs.push_back(road.throughput);
        }
    }
    
    cudaMalloc(&gpu_roadStartIndices, (locationCount + 1) * sizeof(long long));
    cudaMalloc(&gpu_roadEndpoints, roadEndpoints.size() * sizeof(long long));
    cudaMalloc(&gpu_roadDistances, roadDistances.size() * sizeof(long long));
    cudaMalloc(&gpu_roadThroughputs, roadThroughputs.size() * sizeof(long long));
    
    cudaMemcpy(gpu_roadStartIndices, roadStartIndices.data(), (locationCount + 1) * sizeof(long long), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_roadEndpoints, roadEndpoints.data(), roadEndpoints.size() * sizeof(long long), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_roadDistances, roadDistances.data(), roadDistances.size() * sizeof(long long), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_roadThroughputs, roadThroughputs.data(), roadThroughputs.size() * sizeof(long long), cudaMemcpyHostToDevice);
    
    cudaMalloc(&gpu_distances, locationCount * sizeof(long long));
    cudaMalloc(&gpu_predecessors, locationCount * sizeof(long long));
    cudaMalloc(&gpu_processed, locationCount * sizeof(bool));
    
    long long blockCount = (locationCount + THREAD_COUNT - 1) / THREAD_COUNT;
    initializeSearchArrays<<<blockCount, THREAD_COUNT>>>(locationCount, gpu_distances, gpu_predecessors, gpu_processed);
    cudaDeviceSynchronize();
    
    long long zeroDistance = 0;
    cudaMemcpy(gpu_distances + startLocation, &zeroDistance, sizeof(long long), cudaMemcpyHostToDevice);
    
    long long *gpu_blockMinDistances;
    long long *gpu_blockMinNodes;
    cudaMalloc(&gpu_blockMinDistances, blockCount * sizeof(long long));
    cudaMalloc(&gpu_blockMinNodes, blockCount * sizeof(long long));
    
    for (long long iteration = 0; iteration < locationCount; ++iteration) {
        identifyMinimumDistance<<<blockCount, THREAD_COUNT>>>(locationCount, gpu_distances, gpu_processed, 
                                                          gpu_blockMinDistances, gpu_blockMinNodes);
        cudaDeviceSynchronize();
        
        vector<long long> hostBlockMinDistances(blockCount);
        vector<long long> hostBlockMinNodes(blockCount);
        cudaMemcpy(hostBlockMinDistances.data(), gpu_blockMinDistances, blockCount * sizeof(long long), cudaMemcpyDeviceToHost);
        cudaMemcpy(hostBlockMinNodes.data(), gpu_blockMinNodes, blockCount * sizeof(long long), cudaMemcpyDeviceToHost);
        
        long long globalMinDistance = MAX_VALUE;
        long long globalMinNode = -1;
        for (long long i = 0; i < blockCount; ++i) {
            if (hostBlockMinDistances[i] < globalMinDistance) {
                globalMinDistance = hostBlockMinDistances[i];
                globalMinNode = hostBlockMinNodes[i];
            }
        }
        
        if (globalMinNode == -1) break;
        
        processEdges<<<blockCount, THREAD_COUNT>>>(locationCount, gpu_roadStartIndices, gpu_roadEndpoints, 
                                                gpu_roadDistances, gpu_distances, gpu_predecessors, 
                                                gpu_processed, globalMinNode);
        cudaDeviceSynchronize();
    }
    
    if (shelters != nullptr && rankedShelters != nullptr) {
        long long shelterCount = shelters->size();
        
        vector<long long> shelterLocations(shelterCount);
        vector<long long> shelterCapacities(shelterCount);
        
        for (long long i = 0; i < shelterCount; i++) {
            shelterLocations[i] = (*shelters)[i].location;
            shelterCapacities[i] = (*shelters)[i].maxOccupancy;
        }
        
        cudaMalloc(&gpu_shelterLocations, shelterCount * sizeof(long long));
        cudaMalloc(&gpu_shelterCapacities, shelterCount * sizeof(long long));
        float *gpu_shelterScores;
        cudaMalloc(&gpu_shelterScores, shelterCount * sizeof(float));
        
        cudaMemcpy(gpu_shelterLocations, shelterLocations.data(), shelterCount * sizeof(long long), cudaMemcpyHostToDevice);
        cudaMemcpy(gpu_shelterCapacities, shelterCapacities.data(), shelterCount * sizeof(long long), cudaMemcpyHostToDevice);
        
        long long shelterBlocks = (shelterCount + THREAD_COUNT - 1) / THREAD_COUNT;
        computeShelterScores<<<shelterBlocks, THREAD_COUNT>>>(locationCount, shelterCount, gpu_distances,
                                                          gpu_shelterLocations, gpu_shelterCapacities,
                                                          gpu_shelterScores, maxSeniorDistance);
        cudaDeviceSynchronize();
        
        vector<float> shelterScores(shelterCount);
        cudaMemcpy(shelterScores.data(), gpu_shelterScores, shelterCount * sizeof(float), cudaMemcpyDeviceToHost);
        
        rankedShelters->resize(shelterCount);
        for (long long i = 0; i < shelterCount; i++) {
            (*rankedShelters)[i] = {i, shelterScores[i]};
        }
        
        sort(rankedShelters->begin(), rankedShelters->end(), 
             [](const pair<long long, float>& a, const pair<long long, float>& b) {
                 return a.second > b.second;
             });
        
        cudaFree(gpu_shelterLocations);
        cudaFree(gpu_shelterCapacities);
        cudaFree(gpu_shelterScores);
    }
    
    cudaMemcpy(distances.data(), gpu_distances, locationCount * sizeof(long long), cudaMemcpyDeviceToHost);
    cudaMemcpy(predecessors.data(), gpu_predecessors, locationCount * sizeof(long long), cudaMemcpyDeviceToHost);
    
    cudaFree(gpu_roadStartIndices);
    cudaFree(gpu_roadEndpoints);
    cudaFree(gpu_roadDistances);
    cudaFree(gpu_roadThroughputs);
    cudaFree(gpu_distances);
    cudaFree(gpu_predecessors);
    cudaFree(gpu_processed);
    cudaFree(gpu_blockMinDistances);
    cudaFree(gpu_blockMinNodes);
}

// Reconstruct path from predecessors array
void buildPath(long long startLocation, long long endLocation, const vector<long long>& predecessors, vector<long long>& resultPath) {
    resultPath.clear();
    for (long long currentLocation = endLocation; currentLocation != -1; currentLocation = predecessors[currentLocation]) {
        resultPath.push_back(currentLocation);
    }
    reverse(resultPath.begin(), resultPath.end());
}

// Calculate travel time for a group of people on a road
long long calculateMovementTime(long long populationSize, long long roadThroughput, long long roadDistance) {
    long long groupCount = (populationSize + roadThroughput - 1) / roadThroughput;
    long long timePerGroup = roadDistance * 12;
    return timePerGroup * groupCount;
}

// Find road properties between two locations
bool findRoadDetails(const vector<tuple<long long,long long,long long,long long>>& roads, long long source, long long target, 
                    long long& distance, long long& throughput) {
    for (const auto& [u, v, dist, cap] : roads) {
        if ((u == source && v == target) || (u == target && v == source)) {
            distance = dist;
            throughput = cap;
            return true;
        }
    }
    return false;
}

// Check if time limit is approaching
bool timeIsRunningOut(chrono::steady_clock::time_point startTime) {
    auto currentTime = chrono::steady_clock::now();
    auto elapsed = chrono::duration_cast<chrono::milliseconds>(currentTime - startTime).count();
    return elapsed >= MAX_EXECUTION_TIME_MS;
}

// Main evacuation algorithm
void executeEvacuationPlan(
    long long locationCount, long long roadCount, vector<tuple<long long,long long,long long,long long>>& roads,
    long long refugeCount, vector<RefugeLocation>& refuges,
    long long originCount, vector<OriginLocation>& origins,
    long long seniorMobilityLimit,
    long long*& pathSizes, long long**& evacuationPaths, 
    long long*& dropCounts, long long***& populationDrops,
    long long& rescuedCount,
    chrono::steady_clock::time_point startTime
) {
    // Build road network representation
    vector<vector<RoadConnection>> roadNetwork(locationCount);
    for (auto& [u, v, dist, cap] : roads) {
        roadNetwork[u].push_back({v, dist, cap});
        roadNetwork[v].push_back({u, dist, cap});
    }
    
    // Track remaining refuge capacities
    vector<long long> refugeCapacity(locationCount, 0);
    for (auto& refuge : refuges) {
        refugeCapacity[refuge.location] = refuge.maxOccupancy;
    }
    
    // Initialize output arrays
    pathSizes = new long long[originCount];
    evacuationPaths = new long long*[originCount];
    dropCounts = new long long[originCount];
    populationDrops = new long long**[originCount];
    
    // Initialize with default values (everyone stays at origin)
    for (long long i = 0; i < originCount; i++) {
        pathSizes[i] = 1;
        evacuationPaths[i] = new long long[1];
        evacuationPaths[i][0] = origins[i].location;
        
        dropCounts[i] = 1;
        populationDrops[i] = new long long*[1];
        populationDrops[i][0] = new long long[3];
        populationDrops[i][0][0] = origins[i].location;
        populationDrops[i][0][1] = origins[i].youngPopulation;
        populationDrops[i][0][2] = origins[i].seniorPopulation;
    }
    
    // Initialize rescued counter
    rescuedCount = 0;
    
    // Process origins in order of vulnerability (seniors first, then total population)
    vector<long long> originPriority(originCount);
    for (long long i = 0; i < originCount; ++i) {
        originPriority[i] = i;
    }
    
    // Sort by senior population first, then by total population
    sort(originPriority.begin(), originPriority.end(), [&](long long a, long long b) {
        if (origins[a].seniorPopulation != origins[b].seniorPopulation) {
            return origins[a].seniorPopulation > origins[b].seniorPopulation;
        }
        return (origins[a].youngPopulation + origins[a].seniorPopulation) >
               (origins[b].youngPopulation + origins[b].seniorPopulation);
    });
    
    // Road usage timeline
    unordered_map<string, long long> roadAvailabilityTime;
    
    // Process each origin in priority order
    for (long long priorityIdx = 0; priorityIdx < originPriority.size(); ++priorityIdx) {
        // Check time limit periodically
        if (priorityIdx % 3 == 0 && timeIsRunningOut(startTime)) {
            break;
        }
        
        long long originIdx = originPriority[priorityIdx];
        long long originLocation = origins[originIdx].location;
        long long youngPopulation = origins[originIdx].youngPopulation;
        long long seniorPopulation = origins[originIdx].seniorPopulation;
        
        // Find shortest paths and rank shelters
        vector<long long> distances(locationCount);
        vector<long long> predecessors(locationCount);
        vector<RefugeLocation> availableRefuges;
        
        // Collect available refuges
        for (auto& refuge : refuges) {
            if (refugeCapacity[refuge.location] > 0) {
                availableRefuges.push_back(refuge);
            }
        }
        
        vector<pair<long long, float>> rankedRefuges;
        parallelShortestPath(locationCount, roadNetwork, originLocation, distances, predecessors, 
                           seniorMobilityLimit, &availableRefuges, &rankedRefuges);
        
        // Check time again before potentially long operations
        if (timeIsRunningOut(startTime)) {
            break;
        }
        
        // Handle population distribution
        vector<vector<long long>> dropsList;
        long long remainingSeniors = seniorPopulation;
        long long remainingYoung = youngPopulation;
        long long seniorsSaved = 0;
        long long youngSaved = 0;
        
        // First, try to save seniors (they have mobility constraints)
        if (remainingSeniors > 0) {
            // Find all locations within senior mobility range
            vector<pair<long long, long long>> accessibleLocations;
            for (long long loc = 0; loc < locationCount; loc++) {
                if (distances[loc] <= seniorMobilityLimit && distances[loc] != MAX_VALUE) {
                    accessibleLocations.push_back({loc, distances[loc]});
                }
            }
            
            // Sort locations by distance (closest first)
            sort(accessibleLocations.begin(), accessibleLocations.end(),
                 [](const auto& a, const auto& b) { return a.second < b.second; });
            
            // Try to place seniors in refuges within range
            for (const auto& [refugeIdx, score] : rankedRefuges) {
                long long refugeLocation = availableRefuges[refugeIdx].location;
                if (distances[refugeLocation] <= seniorMobilityLimit) {
                    // Ensure we don't exceed capacity and use min() to limit the number of seniors placed
                    long long capacity = min(refugeCapacity[refugeLocation], remainingSeniors);
                    
                    if (capacity > 0) {
                        dropsList.push_back({refugeLocation, 0, capacity});
                        refugeCapacity[refugeLocation] -= capacity;
                        remainingSeniors -= capacity;
                        seniorsSaved += capacity;
                    }
                }
                
                if (remainingSeniors == 0) break;
            }
            
            // If seniors remain, drop at furthest reachable location within mobility limit
            if (remainingSeniors > 0 && !accessibleLocations.empty()) {
                // Sort by distance in descending order to find furthest accessible location
                sort(accessibleLocations.begin(), accessibleLocations.end(),
                     [](const auto& a, const auto& b) { return a.second > b.second; });
                
                for (const auto& [location, dist] : accessibleLocations) {
                    if (dist <= seniorMobilityLimit) {
                        dropsList.push_back({location, 0, remainingSeniors});
                        remainingSeniors = 0;
                        break;
                    }
                }
                
                // If still not placed, keep at origin
                if (remainingSeniors > 0) {
                    dropsList.push_back({originLocation, 0, remainingSeniors});
                    remainingSeniors = 0;
                }
            } else if (remainingSeniors > 0) {
                // If no suitable location found, keep seniors at origin
                dropsList.push_back({originLocation, 0, remainingSeniors});
                remainingSeniors = 0;
            }
        }
        
        // Now handle young population (no distance constraint)
        if (remainingYoung > 0) {
            // Try to place young people in refuges
            for (const auto& [refugeIdx, score] : rankedRefuges) {
                long long refugeLocation = availableRefuges[refugeIdx].location;
                // Ensure we don't exceed capacity and use min() to limit the number of young placed
                long long capacity = min(refugeCapacity[refugeLocation], remainingYoung);
                
                if (capacity > 0) {
                    dropsList.push_back({refugeLocation, capacity, 0});
                    refugeCapacity[refugeLocation] -= capacity;
                    remainingYoung -= capacity;
                    youngSaved += capacity;
                }
                
                if (remainingYoung == 0) break;
            }
            
            // If young people remain, find furthest reachable location
            if (remainingYoung > 0) {
                long long furthestLocation = originLocation;
                long long maxDistance = 0;
                
                for (long long loc = 0; loc < locationCount; loc++) {
                    if (distances[loc] != MAX_VALUE && distances[loc] > maxDistance) {
                        maxDistance = distances[loc];
                        furthestLocation = loc;
                    }
                }
                
                dropsList.push_back({furthestLocation, remainingYoung, 0});
                remainingYoung = 0;
            }
        }
        
        // Update total rescued count
        rescuedCount += youngSaved + seniorsSaved;
        
        // Check time again before another potentially long operation
        if (timeIsRunningOut(startTime)) {
            break;
        }
        
        // Combine drops at the same location - properly combine drops
        unordered_map<long long, vector<long long>> combinedDrops;
        for (auto& drop : dropsList) {
            long long dropLocation = drop[0];
            if (combinedDrops.find(dropLocation) == combinedDrops.end()) {
                combinedDrops[dropLocation] = {dropLocation, 0, 0};
            }
            combinedDrops[dropLocation][1] += drop[1]; // young
            combinedDrops[dropLocation][2] += drop[2]; // seniors
        }
        
        dropsList.clear();
        for (auto& [dropLocation, drop] : combinedDrops) {
            if (drop[1] > 0 || drop[2] > 0) { // Only add drops if there are people
                dropsList.push_back(drop);
            }
        }
        
        // Generate path including all drop points
        vector<long long> dropLocations;
        for (auto& drop : dropsList) {
            dropLocations.push_back(drop[0]);
        }
        
        // Create a path starting from origin location to all drop locations
        vector<long long> evacuationRoute;
        if (!dropLocations.empty()) {
            // Sort drop locations by distance from start
            sort(dropLocations.begin(), dropLocations.end(),
                 [&](long long a, long long b) { return distances[a] < distances[b]; });
                 
            // Start with path to closest drop location
            buildPath(originLocation, dropLocations[0], predecessors, evacuationRoute);
            
            // Add paths to other drop locations
            for (size_t j = 1; j < dropLocations.size(); j++) {
                // Check time before each path calculation
                if (timeIsRunningOut(startTime)) {
                    break;
                }
                
                long long fromLocation = evacuationRoute.back();
                long long toLocation = dropLocations[j];
                
                // Find path from current end to next drop location
                vector<long long> tempDistances(locationCount);
                vector<long long> tempPredecessors(locationCount);
                parallelShortestPath(locationCount, roadNetwork, fromLocation, tempDistances, tempPredecessors);
                
                vector<long long> pathSegment;
                buildPath(fromLocation, toLocation, tempPredecessors, pathSegment);
                
                // Append path (skip first location)
                for (size_t k = 1; k < pathSegment.size(); k++) {
                    evacuationRoute.push_back(pathSegment[k]);
                }
            }
        } else {
            evacuationRoute.push_back(originLocation); // At least include starting location
        }
        
        // Calculate cumulative distances from origin for each city in the path
        vector<long long> cumulativeDistances(evacuationRoute.size(), 0);
        for (size_t i = 1; i < evacuationRoute.size(); i++) {
            long long fromLocation = evacuationRoute[i-1];
            long long toLocation = evacuationRoute[i];
            long long roadDistance = 0;
            long long roadThroughput = 0;
            if (findRoadDetails(roads, fromLocation, toLocation, roadDistance, roadThroughput)) {
                cumulativeDistances[i] = cumulativeDistances[i-1] + roadDistance;
            }
        }
        
        // Sort drops by order in path
        sort(dropsList.begin(), dropsList.end(), [&evacuationRoute](const auto& a, const auto& b) {
            long long locationA = a[0];
            long long locationB = b[0];
            auto itA = find(evacuationRoute.begin(), evacuationRoute.end(), locationA);
            auto itB = find(evacuationRoute.begin(), evacuationRoute.end(), locationB);
            
            return distance(evacuationRoute.begin(), itA) < distance(evacuationRoute.begin(), itB);
        });
        
        // Create a map to track seniors that need to be dropped at each location
        unordered_map<long long, long long> seniorDropsByLocation;
        
        // First, identify where seniors need to be dropped due to mobility constraints
        for (size_t i = 0; i < evacuationRoute.size(); i++) {
            long long location = evacuationRoute[i];
            long long distanceFromOrigin = cumulativeDistances[i];
            
            // If this is the last location seniors can reach, drop them here
            if (distanceFromOrigin <= seniorMobilityLimit && 
                (i == evacuationRoute.size() - 1 || cumulativeDistances[i+1] > seniorMobilityLimit)) {
                
                // Find all seniors that haven't been dropped yet
                long long seniorsToDropHere = 0;
                for (auto& drop : dropsList) {
                    if (find(evacuationRoute.begin() + i + 1, evacuationRoute.end(), drop[0]) != evacuationRoute.end()) {
                        seniorsToDropHere += drop[2];
                        drop[2] = 0; // Remove seniors from later drops
                    }
                }
                
                if (seniorsToDropHere > 0) {
                    seniorDropsByLocation[location] += seniorsToDropHere;
                }
            }
        }
        
        // Now update the drops list with the senior drops
        for (auto& [location, seniorCount] : seniorDropsByLocation) {
            if (seniorCount > 0) {
                // Check if this location already has a drop
                bool found = false;
                for (auto& drop : dropsList) {
                    if (drop[0] == location) {
                        drop[2] += seniorCount; // Add seniors to existing drop
                        found = true;
                        break;
                    }
                }
                
                if (!found) {
                    // Create a new drop for seniors
                    dropsList.push_back({location, 0, seniorCount});
                }
            }
        }
        
        // Re-sort drops by order in path after adding senior drops
        sort(dropsList.begin(), dropsList.end(), [&evacuationRoute](const auto& a, const auto& b) {
            long long locationA = a[0];
            long long locationB = b[0];
            auto itA = find(evacuationRoute.begin(), evacuationRoute.end(), locationA);
            auto itB = find(evacuationRoute.begin(), evacuationRoute.end(), locationB);
            
            return distance(evacuationRoute.begin(), itA) < distance(evacuationRoute.begin(), itB);
        });
        
        // Calculate population at each segment of the path
        vector<long long> totalPopulation(evacuationRoute.size(), youngPopulation + seniorPopulation);
        vector<long long> youngPopulationBySegment(evacuationRoute.size(), youngPopulation);
        vector<long long> seniorPopulationBySegment(evacuationRoute.size(), seniorPopulation);
        
        // Update population counts after each drop
        for (auto& drop : dropsList) {
            long long dropLocation = drop[0];
            auto it = find(evacuationRoute.begin(), evacuationRoute.end(), dropLocation);
            if (it != evacuationRoute.end()) {
                long long dropIndex = distance(evacuationRoute.begin(), it);
                
                // Update population counts after this drop
                for (size_t i = dropIndex + 1; i < evacuationRoute.size(); i++) {
                    totalPopulation[i] -= (drop[1] + drop[2]);
                    youngPopulationBySegment[i] -= drop[1];
                    seniorPopulationBySegment[i] -= drop[2];
                }
            }
        }
        
        // Implement road contention and batch movement
        long long currentTime = 0;
        
        // Calculate travel time for each segment considering road capacity
        for (size_t i = 0; i < evacuationRoute.size() - 1; i++) {
            long long fromLocation = evacuationRoute[i];
            long long toLocation = evacuationRoute[i+1];
            
            // Create a unique key for this road (smaller location ID first)
            string roadKey = fromLocation < toLocation ? 
                to_string(fromLocation) + "-" + to_string(toLocation) : 
                to_string(toLocation) + "-" + to_string(fromLocation);
            
            // Wait if the road is being used
            if (roadAvailabilityTime.find(roadKey) != roadAvailabilityTime.end()) {
                currentTime = max(currentTime, roadAvailabilityTime[roadKey]);
            }
            
            // Find road properties
            long long roadDistance = 0;
            long long roadThroughput = 0;
            if (findRoadDetails(roads, fromLocation, toLocation, roadDistance, roadThroughput)) {
                // Calculate travel time for this segment based on number of people
                long long peopleOnSegment = totalPopulation[i];
                long long segmentTime = calculateMovementTime(peopleOnSegment, roadThroughput, roadDistance);
                
                // Update current time and mark when the road will be free
                roadAvailabilityTime[roadKey] = currentTime + segmentTime;
                currentTime += segmentTime;
            }
        }
        
        // Free previous memory for this origin
        delete[] evacuationPaths[originIdx];
        for (long long j = 0; j < dropCounts[originIdx]; j++) {
            delete[] populationDrops[originIdx][j];
        }
        delete[] populationDrops[originIdx];
        
        // Store results
        pathSizes[originIdx] = evacuationRoute.size();
        evacuationPaths[originIdx] = new long long[evacuationRoute.size()];
        for (size_t j = 0; j < evacuationRoute.size(); j++) {
            evacuationPaths[originIdx][j] = evacuationRoute[j];
        }
        
        // Remove drops with zero people
        vector<vector<long long>> finalDrops;
        for (auto& drop : dropsList) {
            if (drop[1] > 0 || drop[2] > 0) {
                finalDrops.push_back(drop);
            }
        }
        
        dropCounts[originIdx] = finalDrops.size();
        populationDrops[originIdx] = new long long*[finalDrops.size()];
        for (size_t j = 0; j < finalDrops.size(); j++) {
            populationDrops[originIdx][j] = new long long[3];
            populationDrops[originIdx][j][0] = finalDrops[j][0]; 
            populationDrops[originIdx][j][1] = finalDrops[j][1]; 
            populationDrops[originIdx][j][2] = finalDrops[j][2]; 
        }
    }
}

// Function to convert flat array roads to vector of tuples
vector<tuple<long long, long long, long long, long long>> convertRoads(int* roads, long long num_roads) {
    vector<tuple<long long, long long, long long, long long>> roadsVector;
    for (long long i = 0; i < num_roads; i++) {
        roadsVector.push_back(make_tuple(
            roads[4 * i],
            roads[4 * i + 1],
            roads[4 * i + 2],
            roads[4 * i + 3]
        ));
    }
    return roadsVector;
}

//Main function

int main(int argc, char *argv[]) {
    if (argc < 3) {
        cerr << "Usage: " << argv[0] << " <input_file> <output_file>\n";
        return 1;
    }

    ifstream infile(argv[1]);
    if (!infile) {
        cerr << "Error: Cannot open file " << argv[1] << "\n";
        return 1;
    }

    //Input taking from user 
    
    long long num_cities;
    infile >> num_cities;

    long long num_roads;
    infile >> num_roads;
    int *roads = new int[num_roads * 4]; 

    for (int i = 0; i < num_roads; i++) {
        infile >> roads[4 * i] >> roads[4 * i + 1] >> roads[4 * i + 2] >> roads[4 * i + 3];
    }

    int num_shelters;
    infile >> num_shelters;

    long long *shelter_city = new long long[num_shelters];
    long long *shelter_capacity = new long long[num_shelters];

    for (int i = 0; i < num_shelters; i++) {
        infile >> shelter_city[i] >> shelter_capacity[i];
    }

    int num_populated_cities;
    infile >> num_populated_cities;

    long long *city = new long long[num_populated_cities];
    long long *pop = new long long[num_populated_cities * 2];

    for (long long i = 0; i < num_populated_cities; i++) {
        infile >> city[i] >> pop[2 * i] >> pop[2 * i + 1];
    }

    int max_distance_elderly;
    infile >> max_distance_elderly;

    infile.close();

    auto startTime = chrono::steady_clock::now();
    
    vector<tuple<long long, long long, long long, long long>> roadsVector = convertRoads(roads, num_roads);
    
    vector<RefugeLocation> refuges(num_shelters);
    for (int i = 0; i < num_shelters; i++) {
        refuges[i] = {shelter_city[i], shelter_capacity[i]};
    }
    
    vector<OriginLocation> origins(num_populated_cities);
    for (int i = 0; i < num_populated_cities; i++) {
        origins[i] = {city[i], pop[2 * i], pop[2 * i + 1]};
    }
    
    long long *path_size = nullptr;
    long long **paths = nullptr;
    long long *num_drops = nullptr;
    long long ***drops = nullptr;
    long long rescuedCount = 0;
    
    executeEvacuationPlan(
        num_cities, num_roads, roadsVector,
        num_shelters, refuges,
        num_populated_cities, origins,
        max_distance_elderly,
        path_size, paths, num_drops, drops,
        rescuedCount, startTime
    );

    ofstream outfile(argv[2]);
    if (!outfile) {
        cerr << "Error: Cannot open file " << argv[2] << "\n";
        return 1;
    }
    
    for(long long i = 0; i < num_populated_cities; i++) {
        long long currentPathSize = path_size[i];
        for(long long j = 0; j < currentPathSize; j++) {
            outfile << paths[i][j];
            if (j < currentPathSize - 1) outfile << " ";
        }
        outfile << "\n";
    }

    for(long long i = 0; i < num_populated_cities; i++) {
        long long currentDropSize = num_drops[i];
        for(long long j = 0; j < currentDropSize; j++) {
            for(int k = 0; k < 3; k++) {
                outfile << drops[i][j][k] << " ";
            }
        }
        outfile << "\n";
    }
    
    delete[] roads;
    delete[] shelter_city;
    delete[] shelter_capacity;
    delete[] city;
    delete[] pop;
    
    for (long long i = 0; i < num_populated_cities; i++) {
        delete[] paths[i];
        for (long long j = 0; j < num_drops[i]; j++) {
            delete[] drops[i][j];
        }
        delete[] drops[i];
    }
    delete[] path_size;
    delete[] paths;
    delete[] num_drops;
    delete[] drops;

    return 0;
}
