
torch::Tensor ThermodynamicsImpl::get_mole_fraction(torch::Tensor q) const {
  auto qu = q.unfold(0, options.nvapor() + options.ncloud(), 1);
  auto x = qu.matmul(mu_ratio_m1 + 1);
  auto sum = 1. + qu.matmul(mu_ratio_m1);
  return x / sum;
}

torch::Tensor ThermodynamicsImpl::get_mass_fraction(torch::Tensor x) const {
  auto xu = x.unfold(0, options.nvapor() + options.ncloud(), 1);
  auto q = xu.matmul(1. / (mu_ratio_m1 + 1));
  auto sum = 1. - xu.matmul(mu_ratio_m1 / (mu_ratio_m1 + 1.));
  return q / sum;
}

torch::Tensor ThermodynamicsImpl::get_dens(torch::Tensor var, int type) const {
  if (type == kTPMassLR) {
    auto result = torch::zeros_like(var.select(1, 0));
    result[0] = var[0][index::IPR] /
                (var[0][index::IDN] * options.Rd() * f_eps(var[0]));
    result[1] = var[1][index::IPR] /
                (var[1][index::IDN] * options.Rd() * f_eps(var[1]));
    return result;
  } else {
    std::stringstream msg;
    msg << fmt::format("{}::Unknown variable type code: {}", name(), type);
    throw std::runtime_error(msg.str());
  }
}
