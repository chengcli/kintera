
torch::Tensor ThermodynamicsImpl::get_temp(torch::Tensor w) const {
  return w[Index::IPR] / (w[Index::IDN] * options.Rd() * f_eps(w));
}

torch::Tensor ThermodynamicsImpl::get_theta(torch::Tensor w, double p0) const {
  return get_temp(w) * torch::pow(p0 / w[Index::IPR], get_chi(w));
}

torch::Tensor ThermodynamicsImpl::get_chi(torch::Tensor yfrac) const {
  auto gammad = options.gammad_ref();
  return (gammad - 1.) / gammad * f_eps(yfrac) / f_psi(yfrac);
}
