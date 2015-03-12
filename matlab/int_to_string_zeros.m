function str = int_to_string_zeros(num,pre_zeros)
    nz = pre_zeros-floor(log10(num));
    str=[repmat('0',1,nz) num2str(num)];
end
