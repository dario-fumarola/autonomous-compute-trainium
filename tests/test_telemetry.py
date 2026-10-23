import pytest

from scripts.capture_neuron_telemetry import aggregate


def test_per_application_zero_entries_do_not_dilute_device_occupancy():
    data={'neuron_runtime_data':[{'report':{'neuroncore_counters':{'neuroncores_in_use':
        {'0':{'neuroncore_utilization':60},'1':{'neuroncore_utilization':0}}}}},
        {'report':{'neuroncore_counters':{'neuroncores_in_use':
        {'0':{'neuroncore_utilization':0},'1':{'neuroncore_utilization':80}}}}}]}
    result=aggregate(data)
    assert result['active_runtime_processes']==result['active_logical_cores']==2
    assert result['mean_reported_core_busy_percent']==70
    data['neuron_runtime_data'][1]['report']['neuroncore_counters']['neuroncores_in_use']['0']['neuroncore_utilization']=60
    with pytest.raises(ValueError):aggregate(data)
