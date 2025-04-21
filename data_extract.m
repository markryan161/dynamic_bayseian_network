cell1_array = struct2cell(Cell1);

numPoints = 3000;  
numCycles = length(cell1_array);  


temps = NaN(numPoints, numCycles);  
charges = NaN(numPoints, numCycles); 
times = NaN(numPoints, numCycles);
voltages = NaN(numPoints, numCycles);


for i = 1:numCycles
    if isfield(cell1_array{i}, 'C1ch') && isfield(cell1_array{i}.C1ch, 'T')
        tempData = cell1_array{i}.C1ch.T;  

        numAvailable = min(length(tempData), numPoints);
        temps(1:numAvailable, i) = tempData(1:numAvailable);
    end

    if isfield(cell1_array{i}, 'C1ch') && isfield(cell1_array{i}.C1ch, 'q')
        chargeData = cell1_array{i}.C1ch.q;  

        numAvailable = min(length(chargeData), numPoints);
        charges(1:numAvailable, i) = chargeData(1:numAvailable);
    end

    if isfield(cell1_array{i}, 'C1ch') && isfield(cell1_array{i}.C1ch, 't')
        timeData = cell1_array{i}.C1ch.t;  

        numAvailable = min(length(timeData), numPoints);
        times(1:numAvailable, i) = timeData(1:numAvailable);
    end

    if isfield(cell1_array{i}, 'C1ch') && isfield(cell1_array{i}.C1ch, 'v')
        voltageData = cell1_array{i}.C1ch.v;  

        numAvailable = min(length(voltageData), numPoints);
        voltages(1:numAvailable, i) = voltageData(1:numAvailable);
    end
end

max_temps = zeros(78,1);
min_temps = max_temps;
max_charge = max_temps;
min_charge = max_temps;
max_voltages = max_temps;
min_voltages = max_temps;
charging_times = max_temps;


for i = 1:width(temps)
    max_temps(i,1) = max(temps(:,i));
    min_temps(i,1) = min(temps(:,i));
    max_charge(i,1) = max(charges(:,i));
    min_charge(i,1) = min(charges(:,i));
    max_voltages(i,1) = max(voltages(:,i));
    min_voltages(i,1) = min(voltages(:,i))
    charging_times(i,1) = times(1,i);

end


