"""
Performance testing script for the recommendation system.
This script benchmarks the recommendation system under various load conditions.
"""

import sys
import os
import time
import json
import random
import asyncio
import aiohttp
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Configuration
API_URL = 'http://localhost:5000'
NUM_USERS = [1, 5, 10, 20, 50]  # Different user loads to test
NUM_REQUESTS = 20  # Requests per user
NUM_ITERATIONS = 3  # Number of test iterations
CONCURRENCY_LEVELS = [1, 5, 10, 20]  # Number of concurrent requests
EVENT_TYPES = ['view', 'click', 'book', 'rate', 'favorite']

# Result storage
results = {
    'track_event': {},
    'recommend': {},
    'response_times': [],
    'error_rates': [],
    'throughput': []
}

def generate_random_user_id():
    """Generate a random user ID"""
    return f"perf_test_user_{random.randint(1000, 9999)}"

def generate_random_location_id():
    """Generate a random location ID"""
    return random.randint(100, 999)

def generate_random_event():
    """Generate a random event object"""
    user_id = generate_random_user_id()
    location_id = generate_random_location_id()
    event_type = random.choice(EVENT_TYPES)
    
    event_data = {
        'user_id': user_id,
        'location_id': location_id,
        'event_type': event_type,
        'data': {}
    }
    
    # Add rating if event type is 'rate'
    if event_type == 'rate':
        event_data['data']['rating'] = random.randint(1, 5)
    
    return event_data

async def send_event(session, event_data):
    """Send an event to the recommendation system via HTTP API"""
    start_time = time.time()
    success = False
    error = None
    
    try:
        async with session.post(f"{API_URL}/api/track", json=event_data) as response:
            success = response.status == 200
            if not success:
                error = await response.text()
    except Exception as e:
        error = str(e)
    
    elapsed = time.time() - start_time
    return {
        'success': success,
        'elapsed': elapsed,
        'error': error,
        'event_type': event_data['event_type']
    }

async def get_recommendations(session, user_id):
    """Get recommendations for a user"""
    start_time = time.time()
    success = False
    error = None
    num_recommendations = 0
    
    try:
        async with session.get(f"{API_URL}/api/recommend?user_id={user_id}&case=hybrid") as response:
            success = response.status == 200
            if success:
                data = await response.json()
                num_recommendations = len(data.get('recommendations', []))
            else:
                error = await response.text()
    except Exception as e:
        error = str(e)
    
    elapsed = time.time() - start_time
    return {
        'success': success,
        'elapsed': elapsed,
        'error': error,
        'num_recommendations': num_recommendations
    }

async def run_user_load(num_users, concurrency, endpoint='track_event'):
    """Run a load test with a specified number of users and concurrency level"""
    async with aiohttp.ClientSession() as session:
        all_tasks = []
        
        # Generate all the tasks upfront
        for _ in range(num_users):
            user_id = generate_random_user_id()
            
            for _ in range(NUM_REQUESTS):
                if endpoint == 'track_event':
                    event_data = generate_random_event()
                    all_tasks.append(send_event(session, event_data))
                elif endpoint == 'recommend':
                    all_tasks.append(get_recommendations(session, user_id))
        
        # Process tasks in batches based on concurrency
        all_results = []
        for i in range(0, len(all_tasks), concurrency):
            batch = all_tasks[i:i+concurrency]
            batch_results = await asyncio.gather(*batch)
            all_results.extend(batch_results)
        
        return all_results

async def run_performance_test():
    """Run the full performance test suite"""
    print(f"Starting performance tests with {NUM_ITERATIONS} iterations")
    
    # Test different endpoints
    for endpoint in ['track_event', 'recommend']:
        print(f"\nTesting endpoint: {endpoint}")
        results[endpoint]['by_users'] = {}
        results[endpoint]['by_concurrency'] = {}
        
        # Test different user loads
        for num_users in NUM_USERS:
            print(f"  Testing with {num_users} users...")
            results[endpoint]['by_users'][num_users] = []
            
            for concurrency in CONCURRENCY_LEVELS:
                print(f"    Concurrency level: {concurrency}")
                iteration_results = []
                
                # Run multiple iterations for statistical significance
                for i in range(NUM_ITERATIONS):
                    start_time = time.time()
                    test_results = await run_user_load(num_users, concurrency, endpoint)
                    total_time = time.time() - start_time
                    
                    # Calculate metrics
                    success_count = sum(1 for r in test_results if r['success'])
                    error_rate = (len(test_results) - success_count) / len(test_results) if test_results else 0
                    avg_response_time = np.mean([r['elapsed'] for r in test_results]) if test_results else 0
                    throughput = len(test_results) / total_time if total_time > 0 else 0
                    
                    iteration_results.append({
                        'iteration': i + 1,
                        'success_count': success_count,
                        'error_rate': error_rate,
                        'avg_response_time': avg_response_time,
                        'total_time': total_time,
                        'throughput': throughput
                    })
                
                # Store results for this concurrency level
                if concurrency not in results[endpoint]['by_concurrency']:
                    results[endpoint]['by_concurrency'][concurrency] = []
                
                # Calculate average metrics across iterations
                avg_metrics = {
                    'num_users': num_users,
                    'concurrency': concurrency,
                    'error_rate': np.mean([r['error_rate'] for r in iteration_results]),
                    'avg_response_time': np.mean([r['avg_response_time'] for r in iteration_results]),
                    'throughput': np.mean([r['throughput'] for r in iteration_results])
                }
                
                results[endpoint]['by_users'][num_users].append(avg_metrics)
                results[endpoint]['by_concurrency'][concurrency].append(avg_metrics)
                
                # Track overall metrics
                results['response_times'].append(avg_metrics['avg_response_time'])
                results['error_rates'].append(avg_metrics['error_rate'])
                results['throughput'].append(avg_metrics['throughput'])
    
    # Save results
    save_results()
    
    # Generate visualizations
    generate_visualizations()
    
    print("\nPerformance tests completed!")

def save_results():
    """Save test results to file"""
    filename = f"performance_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    # Convert numpy values to Python native types
    results_copy = json.loads(json.dumps(results, default=lambda x: float(x) if isinstance(x, np.number) else x))
    
    with open(filename, 'w') as f:
        json.dump(results_copy, f, indent=2)
    
    print(f"Results saved to {filename}")

def generate_visualizations():
    """Generate visualizations of test results"""
    try:
        # Set up the figure
        plt.figure(figsize=(15, 10))
        
        # Plot 1: Response time by user load
        plt.subplot(2, 2, 1)
        for endpoint in ['track_event', 'recommend']:
            x = []
            y = []
            for num_users, metrics_list in results[endpoint]['by_users'].items():
                for metrics in metrics_list:
                    x.append(num_users)
                    y.append(metrics['avg_response_time'])
            plt.scatter(x, y, alpha=0.7, label=endpoint)
        
        plt.xlabel('Number of Users')
        plt.ylabel('Average Response Time (s)')
        plt.title('Response Time by User Load')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Plot 2: Error rate by concurrency
        plt.subplot(2, 2, 2)
        for endpoint in ['track_event', 'recommend']:
            x = []
            y = []
            for concurrency, metrics_list in results[endpoint]['by_concurrency'].items():
                for metrics in metrics_list:
                    x.append(concurrency)
                    y.append(metrics['error_rate'] * 100)  # Convert to percentage
            plt.scatter(x, y, alpha=0.7, label=endpoint)
        
        plt.xlabel('Concurrency Level')
        plt.ylabel('Error Rate (%)')
        plt.title('Error Rate by Concurrency')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Plot 3: Throughput by concurrency
        plt.subplot(2, 2, 3)
        for endpoint in ['track_event', 'recommend']:
            x = []
            y = []
            for concurrency, metrics_list in results[endpoint]['by_concurrency'].items():
                for metrics in metrics_list:
                    x.append(concurrency)
                    y.append(metrics['throughput'])
            plt.scatter(x, y, alpha=0.7, label=endpoint)
        
        plt.xlabel('Concurrency Level')
        plt.ylabel('Throughput (req/s)')
        plt.title('Throughput by Concurrency')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Plot 4: Response time distribution
        plt.subplot(2, 2, 4)
        for endpoint in ['track_event', 'recommend']:
            response_times = []
            for metrics_list in results[endpoint]['by_users'].values():
                for metrics in metrics_list:
                    response_times.append(metrics['avg_response_time'])
            plt.hist(response_times, alpha=0.5, label=endpoint, bins=20)
        
        plt.xlabel('Response Time (s)')
        plt.ylabel('Frequency')
        plt.title('Response Time Distribution')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Adjust layout and save
        plt.tight_layout()
        plt.savefig(f"performance_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        plt.close()
        
        print("Visualization saved")
    except Exception as e:
        print(f"Error generating visualizations: {str(e)}")

if __name__ == "__main__":
    print("Recommendation System Performance Test")
    print("=====================================")
    
    # Check server availability
    try:
        import requests
        response = requests.get(f"{API_URL}/api/recommend?user_id=1&case=hybrid")
        if response.status_code == 200:
            print(f"Server is available at {API_URL}")
        else:
            print(f"Server returned unexpected status code: {response.status_code}")
            sys.exit(1)
    except Exception as e:
        print(f"Server is not available at {API_URL}: {str(e)}")
        print("Please make sure the recommendation server is running")
        sys.exit(1)
    
    # Run async performance test
    asyncio.run(run_performance_test())
