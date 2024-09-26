% /Applications/MATLAB_R2021b.app/bin/matlab -nojvm -nodesktop -batch "get_osc_data_ephys_tracked"
wind = 2^12;
step=2^9;
FS=1000;
acorr_bin = 0.02; 
cntl_lo = 250; 
cntl_hi = 500; 
min_rate = 5; 
min_length = 30;
max_n = 7; 
psd_threshp = 0.05; 
phase_threshp = 0.05;

beta_srch_lo = 12; beta_srch_hi = 30; beta_force_freq = 21;
delta_srch_lo = 0.5; delta_srch_hi = 4; delta_force_freq = 2;

freqs = 0:FS/wind:50; % for most purposes only care about this range
freqs_long = 0:FS/wind:FS/2-FS/wind; % full range

direcs = ["npas_neurons/tracked_units","jaws_neurons/tracked_units"];

for direc = 1:length(direcs)
    cell_files = dir(fullfile(sprintf('data/tracked_units/%s/cell_*'),direcs(direc)));
    for k = 1:length(cell_files)
        if contains(cell_files(k).name,"cell")
            spike_files = dir(fullfile(sprintf('data/tracked_units/%s/%s/spikes/segment*.txt',direcs(direc),cell_files(k).name)));
            firing_rates = importdata(sprintf('data/tracked_units/%s/%s/rates.txt',direcs(direc),cell_files(k).name));
            recording_lengths = ones(length(firing_rates),1)*min_length;
            output_data = zeros(length(spike_files),8); %  (1) delta osc, (2) renewal power, (3) reg power, (4) peak_freq, (5) beta osc, (6) renewal power, (7) reg power, (8) peak freq

            for i = 1:length(spike_files)
                if firing_rates(i) >= min_rate && recording_lengths(i) >= min_length
                    spikes = importdata(sprintf("data/tracked_units/%s/%s/spikes/%s",direcs(direc),cell_files(k).name,spike_files(i).name));

                    % Check delta osc
                    [~, delta_freq_ind] = min(abs(freqs-delta_force_freq));
                    [psd_corr, phshift, psd_unc] = renewalPSD_phaseShift(spikes, 'wind', wind, 'step', step, 'FS', FS);
                    srch_inds = [find(freqs_long>=delta_srch_lo,1), find(freqs_long<=delta_srch_hi,1,'last')];
                    cntl_inds = [find(freqs_long>cntl_lo,1), find(freqs_long<=cntl_hi,1,'last')];
                    [sigp_inds,sig_inds] = find_sig_osc(psd_corr,phshift,srch_inds,cntl_inds,max_n,psd_threshp,phase_threshp);
                    [max1,I1] = max(psd_corr);
                    output_data(i,2) = psd_corr(delta_freq_ind)/firing_rates(i);% renewal power
                    output_data(i,3) = psd_unc(delta_freq_ind)/firing_rates(i);%  power
                    output_data(i,4) = freqs_long(I1) ;% peak freq
                    if ~isempty(sigp_inds)
                        output_data(i,1) = 1;
                    end

                    % Check beta osc
                    [~, beta_freq_ind] = min(abs(freqs-beta_force_freq));
                    [psd_corr, phshift, psd_unc] = renewalPSD_phaseShift(spikes, 'wind', wind, 'step', step, 'FS', FS);
                    srch_inds = [find(freqs_long>=beta_srch_lo,1), find(freqs_long<=beta_srch_hi,1,'last')];
                    cntl_inds = [find(freqs_long>cntl_lo,1), find(freqs_long<=cntl_hi,1,'last')];
                    [sigp_inds,sig_inds] = find_sig_osc(psd_corr,phshift,srch_inds,cntl_inds,max_n,psd_threshp,phase_threshp);
                    [max1,I1] = max(psd_corr);
                    output_data(i,6) = psd_corr(beta_freq_ind)/firing_rates(i);% renewal power
                    output_data(i,7) = psd_unc(beta_freq_ind)/firing_rates(i);% power
                    output_data(i,8) = freqs_long(I1) ;% peak freq
                    if ~isempty(sigp_inds)
                        output_data(i,5) = 1;
                    end
                else
                    output_data(i,:) = zeros(1,8);
                end
            end


            writematrix(output_data,sprintf("data/tracked_units/%s/%s/osc_data.txt",direcs(direc),cell_files(k).name));

        end
    end
end



exit
recording_lengths = importdata(sprintf('data/tracked_units/%s/cell_lengths.txt',direcs(direc)));
firing_rates = importdata(sprintf('data/tracked_units/%s/cell_frs.txt',direcs(direc)));

output_data = zeros(length(spike_files),6); %  delta osc, power, peak_freq, beta osc, power, peak freq

for i = 1:length(spike_files)
    if firing_rates(i) >= min_rate && recording_lengths(i) >= min_length
        spikes = importdata(sprintf("data/tracked_units/%s/spikes/%s",direcs(direc),spike_files(i).name));

        % Check delta osc
        [~, delta_freq_ind] = min(abs(freqs-delta_force_freq));
        [psd_corr, phshift, psd_unc] = renewalPSD_phaseShift(spikes, 'wind', wind, 'step', step, 'FS', FS);
        srch_inds = [find(freqs_long>=delta_srch_lo,1), find(freqs_long<=delta_srch_hi,1,'last')];
        cntl_inds = [find(freqs_long>cntl_lo,1), find(freqs_long<=cntl_hi,1,'last')];
        [sigp_inds,sig_inds] = find_sig_osc(psd_corr,phshift,srch_inds,cntl_inds,max_n,psd_threshp,phase_threshp);
        [max1,I1] = max(psd_corr);
        output_data(i,2) = psd_unc(delta_freq_ind)/firing_rates(i);% power
        output_data(i,3) = freqs_long(I1) ;% peak freq
        if ~isempty(sigp_inds)
            output_data(i,1) = 1;
        end

        % Check beta osc
        [~, beta_freq_ind] = min(abs(freqs-beta_force_freq));
        [psd_corr, phshift, psd_unc] = renewalPSD_phaseShift(spikes, 'wind', wind, 'step', step, 'FS', FS);
        srch_inds = [find(freqs_long>=beta_srch_lo,1), find(freqs_long<=beta_srch_hi,1,'last')];
        cntl_inds = [find(freqs_long>cntl_lo,1), find(freqs_long<=cntl_hi,1,'last')];
        [sigp_inds,sig_inds] = find_sig_osc(psd_corr,phshift,srch_inds,cntl_inds,max_n,psd_threshp,phase_threshp);
        [max1,I1] = max(psd_corr);
        output_data(i,5) = psd_unc(beta_freq_ind)/firing_rates(i);% power
        output_data(i,6) = freqs_long(I1) ;% peak freq
        if ~isempty(sigp_inds)
            output_data(i,4) = 1;
        end
    else
        output_data(i,:) = zeros(1,6);
    end
end


writematrix(output_data,sprintf("data/tracked_units/%s/osc_data.txt",direcs(direc)));


