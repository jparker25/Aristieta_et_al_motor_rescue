% /Applications/MATLAB_R2021b.app/bin/matlab -nojvm -nodesktop -batch "get_osc_data_pulse_cont"
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

direcs = ["npas_uni_dd_neurons/pre_opto","pv_bilateral_dd_neurons/pre_opto","npas_uni_dd_neurons/post_opto","pv_bilateral_dd_neurons/post_opto"];

for direc = 1:length(direcs)
    spike_files = dir(fullfile(sprintf('../resub_data/%s/spikes/*.txt',direcs(direc))));
    recording_lengths = importdata(sprintf('../resub_data/%s/cell_lengths.txt',direcs(direc)));
    firing_rates = importdata(sprintf('../resub_data/%s/cell_frs.txt',direcs(direc)));
    firing_rates(1)
    
    output_data = zeros(length(spike_files),8); %  (1) delta osc, (2) renewal power, (3) reg power, (4) peak_freq, (5) beta osc, (6) renewal power, (7) reg power, (8) peak freq

    for i = 1:length(spike_files)
        i
        if i == 31
            i*2
        end
        if firing_rates(i) >= min_rate && recording_lengths(i) >= min_length
            spikes = importdata(sprintf("../resub_data/%s/spikes/%s",direcs(direc),spike_files(i).name));
            % Check delta osc
            [~, delta_freq_ind] = min(abs(freqs-delta_force_freq));
            [psd_corr, phshift, psd_unc] = renewalPSD_phaseShift(spikes, 'wind', wind, 'step', step, 'FS', FS);
            srch_inds = [find(freqs_long>=delta_srch_lo,1), find(freqs_long<=delta_srch_hi,1,'last')];
            cntl_inds = [find(freqs_long>cntl_lo,1), find(freqs_long<=cntl_hi,1,'last')];
            [sigp_inds,sig_inds] = find_sig_osc(psd_corr,phshift,srch_inds,cntl_inds,max_n,psd_threshp,phase_threshp);
            [max1,I1] = max(psd_corr);
            output_data(i,2) = psd_corr(delta_freq_ind)/firing_rates(i);% renewal power
            output_data(i,3) = psd_unc(delta_freq_ind)/firing_rates(i);% renewal power
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
            output_data(i,6) = psd_corr(beta_freq_ind)/firing_rates(i);% power
            output_data(i,7) = psd_unc(beta_freq_ind)/firing_rates(i);% power
            output_data(i,8) = freqs_long(I1) ;% peak freq
            if ~isempty(sigp_inds)
                output_data(i,5) = 1;
            end
        else
            output_data(i,:) = zeros(1,8);
        end
    end


    writematrix(output_data,sprintf("../resub_data/%s/osc_data.txt",direcs(direc)));
end

